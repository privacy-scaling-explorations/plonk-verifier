use crate::{
    loader::{
        evm::{
            code::{Precompiled, SolidityAssemblyCode},
            fe_to_u256, modulus, u256_to_fe, U256, U512,
        },
        EcPointLoader, LoadedEcPoint, LoadedScalar, Loader, ScalarLoader,
    },
    util::{
        arithmetic::{CurveAffine, FieldOps, PrimeField},
        Itertools,
    },
};
use hex;
use std::{
    cell::RefCell,
    collections::HashMap,
    fmt::{self, Debug},
    iter,
    ops::{Add, AddAssign, DerefMut, Mul, MulAssign, Neg, Sub, SubAssign},
    rc::Rc,
};

/// Memory pointer starts at 0x80, which is the end of the Solidity memory layout scratch space.
pub const MEM_PTR_START: usize = 0x80;

#[derive(Clone, Debug)]
pub enum Value<T> {
    Constant(T),
    Memory(usize),
    Negated(Box<Value<T>>),
    Sum(Box<Value<T>>, Box<Value<T>>),
    Product(Box<Value<T>>, Box<Value<T>>),
}

impl<T: Debug> PartialEq for Value<T> {
    fn eq(&self, other: &Self) -> bool {
        self.identifier() == other.identifier()
    }
}

impl<T: Debug> Value<T> {
    fn identifier(&self) -> String {
        match self {
            Value::Constant(_) | Value::Memory(_) => format!("{self:?}"),
            Value::Negated(value) => format!("-({value:?})"),
            Value::Sum(lhs, rhs) => format!("({lhs:?} + {rhs:?})"),
            Value::Product(lhs, rhs) => format!("({lhs:?} * {rhs:?})"),
        }
    }
}

/// `Loader` implementation for generating yul code as EVM verifier.
#[derive(Clone, Debug)]
pub struct EvmLoader {
    base_modulus: U256,
    scalar_modulus: U256,
    code: RefCell<SolidityAssemblyCode>,
    ptr: RefCell<usize>,
    cache: RefCell<HashMap<String, usize>>,
}

fn hex_encode_u256(value: &U256) -> String {
    format!("0x{}", hex::encode(value.to_be_bytes::<32>()))
}

impl EvmLoader {
    /// Initialize a [`EvmLoader`] with base and scalar field.
    pub fn new<Base, Scalar>() -> Rc<Self>
    where
        Base: PrimeField<Repr = [u8; 0x20]>,
        Scalar: PrimeField<Repr = [u8; 32]>,
    {
        let base_modulus = modulus::<Base>();
        let scalar_modulus = modulus::<Scalar>();
        let code = SolidityAssemblyCode::new();

        Rc::new(Self {
            base_modulus,
            scalar_modulus,
            code: RefCell::new(code),
            ptr: RefCell::new(MEM_PTR_START),
            cache: Default::default(),
        })
    }

    /// Returns generated Solidity code. This is "Solidity" code that is wrapped in an assembly block.
    /// In other words, it's basically just assembly (equivalently, Yul).
    pub fn solidity_code(self: &Rc<Self>) -> String {
        let code = "
            // Revert if anything fails
            if iszero(success) { revert(0, 0) }

            // Return empty bytes on success
            return(0, 0)"
            .to_string();
        self.code.borrow_mut().runtime_append(code);
        self.code
            .borrow()
            .code(hex_encode_u256(&self.base_modulus), hex_encode_u256(&self.scalar_modulus))
    }

    /// Allocates memory chunk with given `size` and returns pointer.
    pub fn allocate(self: &Rc<Self>, size: usize) -> usize {
        let ptr = *self.ptr.borrow();
        *self.ptr.borrow_mut() += size;
        ptr
    }

    pub(crate) fn ptr(&self) -> usize {
        *self.ptr.borrow()
    }

    pub(crate) fn code_mut(&self) -> impl DerefMut<Target = SolidityAssemblyCode> + '_ {
        self.code.borrow_mut()
    }

    fn push(self: &Rc<Self>, scalar: &Scalar) -> String {
        match scalar.value.clone() {
            Value::Constant(constant) => {
                format!("{constant}")
            }
            Value::Memory(ptr) => {
                format!("mload({ptr:#x})")
            }
            Value::Negated(value) => {
                let v = self.push(&self.scalar(*value));
                format!("sub(f_q, {v})")
            }
            Value::Sum(lhs, rhs) => {
                let lhs = self.push(&self.scalar(*lhs));
                let rhs = self.push(&self.scalar(*rhs));
                format!("addmod({lhs}, {rhs}, f_q)")
            }
            Value::Product(lhs, rhs) => {
                let lhs = self.push(&self.scalar(*lhs));
                let rhs = self.push(&self.scalar(*rhs));
                format!("mulmod({lhs}, {rhs}, f_q)")
            }
        }
    }

    /// Calldata load a field element.
    pub fn calldataload_scalar(self: &Rc<Self>, offset: usize) -> Scalar {
        let ptr = self.allocate(0x20);
        let code = format!("mstore({ptr:#x}, mod(calldataload({offset:#x}), f_q))");
        self.code.borrow_mut().runtime_append(code);
        self.scalar(Value::Memory(ptr))
    }

    /// Calldata load an elliptic curve point and validate it's on affine plane.
    /// Note that identity will cause the verification to fail.
    pub fn calldataload_ec_point(self: &Rc<Self>, offset: usize) -> EcPoint {
        let x_ptr = self.allocate(0x40);
        let y_ptr = x_ptr + 0x20;
        let x_cd_ptr = offset;
        let y_cd_ptr = offset + 0x20;
        let validate_code = self.validate_ec_point();
        let code = format!(
            "
        {{
            let x := calldataload({x_cd_ptr:#x})
            mstore({x_ptr:#x}, x)
            let y := calldataload({y_cd_ptr:#x})
            mstore({y_ptr:#x}, y)
            {validate_code}
        }}"
        );
        self.code.borrow_mut().runtime_append(code);
        self.ec_point(Value::Memory(x_ptr))
    }

    /// Decode an elliptic curve point from limbs.
    pub fn ec_point_from_limbs<const LIMBS: usize, const BITS: usize>(
        self: &Rc<Self>,
        x_limbs: [&Scalar; LIMBS],
        y_limbs: [&Scalar; LIMBS],
    ) -> EcPoint {
        let ptr = self.allocate(0x40);
        let mut code = String::new();
        for (idx, limb) in x_limbs.iter().enumerate() {
            let limb_i = self.push(limb);
            let shift = idx * BITS;
            if idx == 0 {
                code.push_str(format!("let x := {limb_i}\n").as_str());
            } else {
                code.push_str(format!("x := add(x, shl({shift}, {limb_i}))\n").as_str());
            }
        }
        let x_ptr = ptr;
        code.push_str(format!("mstore({x_ptr}, x)\n").as_str());
        for (idx, limb) in y_limbs.iter().enumerate() {
            let limb_i = self.push(limb);
            let shift = idx * BITS;
            if idx == 0 {
                code.push_str(format!("let y := {limb_i}\n").as_str());
            } else {
                code.push_str(format!("y := add(y, shl({shift}, {limb_i}))\n").as_str());
            }
        }
        let y_ptr = ptr + 0x20;
        code.push_str(format!("mstore({y_ptr}, y)\n").as_str());
        let validate_code = self.validate_ec_point();
        let code = format!(
            "{{
            {code}
            {validate_code}
        }}"
        );
        self.code.borrow_mut().runtime_append(code);
        self.ec_point(Value::Memory(ptr))
    }

    fn validate_ec_point(self: &Rc<Self>) -> String {
        "success := and(validate_ec_point(x, y), success)".to_string()
    }

    pub(crate) fn scalar(self: &Rc<Self>, value: Value<U256>) -> Scalar {
        let value = if matches!(value, Value::Constant(_) | Value::Memory(_) | Value::Negated(_)) {
            value
        } else {
            let identifier = value.identifier();
            let some_ptr = self.cache.borrow().get(&identifier).cloned();
            let ptr = if let Some(ptr) = some_ptr {
                ptr
            } else {
                let v = self.push(&Scalar { loader: self.clone(), value });
                let ptr = self.allocate(0x20);
                self.code.borrow_mut().runtime_append(format!("mstore({ptr:#x}, {v})"));
                self.cache.borrow_mut().insert(identifier, ptr);
                ptr
            };
            Value::Memory(ptr)
        };
        Scalar { loader: self.clone(), value }
    }

    fn ec_point(self: &Rc<Self>, value: Value<(U256, U256)>) -> EcPoint {
        EcPoint { loader: self.clone(), value }
    }

    /// Performs `KECCAK256` on `memory[ptr..ptr+len]` and returns pointer of
    /// hash.
    pub fn keccak256(self: &Rc<Self>, ptr: usize, len: usize) -> usize {
        let hash_ptr = self.allocate(0x20);
        let code = format!("mstore({hash_ptr:#x}, keccak256({ptr:#x}, {len}))");
        self.code.borrow_mut().runtime_append(code);
        hash_ptr
    }

    /// Copies a field element into given `ptr`.
    pub fn copy_scalar(self: &Rc<Self>, scalar: &Scalar, ptr: usize) {
        let scalar = self.push(scalar);
        self.code.borrow_mut().runtime_append(format!("mstore({ptr:#x}, {scalar})"));
    }

    /// Allocates a new field element and copies the given value into it.
    pub fn dup_scalar(self: &Rc<Self>, scalar: &Scalar) -> Scalar {
        let ptr = self.allocate(0x20);
        self.copy_scalar(scalar, ptr);
        self.scalar(Value::Memory(ptr))
    }

    /// Allocates a new elliptic curve point and copies the given value into it.
    pub fn dup_ec_point(self: &Rc<Self>, value: &EcPoint) -> EcPoint {
        let ptr = self.allocate(0x40);
        match value.value {
            Value::Constant((x, y)) => {
                let x_ptr = ptr;
                let y_ptr = ptr + 0x20;
                let x = hex_encode_u256(&x);
                let y = hex_encode_u256(&y);
                let code = format!(
                    "mstore({x_ptr:#x}, {x})
                    mstore({y_ptr:#x}, {y})"
                );
                self.code.borrow_mut().runtime_append(code);
            }
            Value::Memory(src_ptr) => {
                let x_ptr = ptr;
                let y_ptr = ptr + 0x20;
                let src_x = src_ptr;
                let src_y = src_ptr + 0x20;
                let code = format!(
                    "mstore({x_ptr:#x}, mload({src_x:#x}))
                    mstore({y_ptr:#x}, mload({src_y:#x}))"
                );
                self.code.borrow_mut().runtime_append(code);
            }
            Value::Negated(_) | Value::Sum(_, _) | Value::Product(_, _) => {
                unreachable!()
            }
        }
        self.ec_point(Value::Memory(ptr))
    }

    fn staticcall(self: &Rc<Self>, precompile: Precompiled, cd_ptr: usize, rd_ptr: usize) {
        let (cd_len, rd_len) = match precompile {
            Precompiled::BigModExp => (0xc0, 0x20),
            Precompiled::Bn254Add => (0x80, 0x40),
            Precompiled::Bn254ScalarMul => (0x60, 0x40),
            Precompiled::Bn254Pairing => (0x180, 0x20),
        };
        let a = precompile as usize;
        let code = format!("success := and(eq(staticcall(gas(), {a:#x}, {cd_ptr:#x}, {cd_len:#x}, {rd_ptr:#x}, {rd_len:#x}), 1), success)");
        self.code.borrow_mut().runtime_append(code);
    }

    fn invert(self: &Rc<Self>, scalar: &Scalar) -> Scalar {
        let rd_ptr = self.allocate(0x20);
        let [cd_ptr, ..] = [
            &self.scalar(Value::Constant(U256::from(0x20))),
            &self.scalar(Value::Constant(U256::from(0x20))),
            &self.scalar(Value::Constant(U256::from(0x20))),
            scalar,
            &self.scalar(Value::Constant(self.scalar_modulus - U256::from(2))),
            &self.scalar(Value::Constant(self.scalar_modulus)),
        ]
        .map(|value| self.dup_scalar(value).ptr());
        self.staticcall(Precompiled::BigModExp, cd_ptr, rd_ptr);
        self.scalar(Value::Memory(rd_ptr))
    }

    fn ec_point_add(self: &Rc<Self>, lhs: &EcPoint, rhs: &EcPoint) -> EcPoint {
        let rd_ptr = self.dup_ec_point(lhs).ptr();
        self.dup_ec_point(rhs);
        self.staticcall(Precompiled::Bn254Add, rd_ptr, rd_ptr);
        self.ec_point(Value::Memory(rd_ptr))
    }

    fn ec_point_scalar_mul(self: &Rc<Self>, ec_point: &EcPoint, scalar: &Scalar) -> EcPoint {
        let rd_ptr = self.dup_ec_point(ec_point).ptr();
        self.dup_scalar(scalar);
        self.staticcall(Precompiled::Bn254ScalarMul, rd_ptr, rd_ptr);
        self.ec_point(Value::Memory(rd_ptr))
    }

    /// Performs pairing.
    pub fn pairing(
        self: &Rc<Self>,
        lhs: &EcPoint,
        g2: (U256, U256, U256, U256),
        rhs: &EcPoint,
        minus_s_g2: (U256, U256, U256, U256),
    ) {
        let rd_ptr = self.dup_ec_point(lhs).ptr();
        self.allocate(0x80);
        let g2_0 = hex_encode_u256(&g2.0);
        let g2_0_ptr = rd_ptr + 0x40;
        let g2_1 = hex_encode_u256(&g2.1);
        let g2_1_ptr = rd_ptr + 0x60;
        let g2_2 = hex_encode_u256(&g2.2);
        let g2_2_ptr = rd_ptr + 0x80;
        let g2_3 = hex_encode_u256(&g2.3);
        let g2_3_ptr = rd_ptr + 0xa0;
        let code = format!(
            "mstore({g2_0_ptr:#x}, {g2_0})
            mstore({g2_1_ptr:#x}, {g2_1})
            mstore({g2_2_ptr:#x}, {g2_2})
            mstore({g2_3_ptr:#x}, {g2_3})"
        );
        self.code.borrow_mut().runtime_append(code);
        self.dup_ec_point(rhs);
        self.allocate(0x80);
        let minus_s_g2_0 = hex_encode_u256(&minus_s_g2.0);
        let minus_s_g2_0_ptr = rd_ptr + 0x100;
        let minus_s_g2_1 = hex_encode_u256(&minus_s_g2.1);
        let minus_s_g2_1_ptr = rd_ptr + 0x120;
        let minus_s_g2_2 = hex_encode_u256(&minus_s_g2.2);
        let minus_s_g2_2_ptr = rd_ptr + 0x140;
        let minus_s_g2_3 = hex_encode_u256(&minus_s_g2.3);
        let minus_s_g2_3_ptr = rd_ptr + 0x160;
        let code = format!(
            "mstore({minus_s_g2_0_ptr:#x}, {minus_s_g2_0})
            mstore({minus_s_g2_1_ptr:#x}, {minus_s_g2_1})
            mstore({minus_s_g2_2_ptr:#x}, {minus_s_g2_2})
            mstore({minus_s_g2_3_ptr:#x}, {minus_s_g2_3})"
        );
        self.code.borrow_mut().runtime_append(code);
        self.staticcall(Precompiled::Bn254Pairing, rd_ptr, rd_ptr);
        let code = format!("success := and(eq(mload({rd_ptr:#x}), 1), success)");
        self.code.borrow_mut().runtime_append(code);
    }

    fn add(self: &Rc<Self>, lhs: &Scalar, rhs: &Scalar) -> Scalar {
        if let (Value::Constant(lhs), Value::Constant(rhs)) = (&lhs.value, &rhs.value) {
            let out = (U512::from(*lhs) + U512::from(*rhs)) % U512::from(self.scalar_modulus);
            return self.scalar(Value::Constant(U256::from(out)));
        }

        self.scalar(Value::Sum(Box::new(lhs.value.clone()), Box::new(rhs.value.clone())))
    }

    fn sub(self: &Rc<Self>, lhs: &Scalar, rhs: &Scalar) -> Scalar {
        if rhs.is_const() {
            return self.add(lhs, &self.neg(rhs));
        }

        self.scalar(Value::Sum(
            Box::new(lhs.value.clone()),
            Box::new(Value::Negated(Box::new(rhs.value.clone()))),
        ))
    }

    fn mul(self: &Rc<Self>, lhs: &Scalar, rhs: &Scalar) -> Scalar {
        if let (Value::Constant(lhs), Value::Constant(rhs)) = (&lhs.value, &rhs.value) {
            let out = (U512::from(*lhs) * U512::from(*rhs)) % U512::from(self.scalar_modulus);
            return self.scalar(Value::Constant(U256::from(out)));
        }

        self.scalar(Value::Product(Box::new(lhs.value.clone()), Box::new(rhs.value.clone())))
    }

    fn neg(self: &Rc<Self>, scalar: &Scalar) -> Scalar {
        if let Value::Constant(constant) = scalar.value {
            return self.scalar(Value::Constant(self.scalar_modulus - constant));
        }

        self.scalar(Value::Negated(Box::new(scalar.value.clone())))
    }
}

#[cfg(test)]
impl EvmLoader {
    fn start_gas_metering(self: &Rc<Self>, _: &str) {
        //  unimplemented
    }

    fn end_gas_metering(self: &Rc<Self>) {
        //  unimplemented
    }

    #[allow(dead_code)]
    fn print_gas_metering(self: &Rc<Self>, _: Vec<u64>) {
        //  unimplemented
    }
}

/// Elliptic curve point.
#[derive(Clone)]
pub struct EcPoint {
    loader: Rc<EvmLoader>,
    value: Value<(U256, U256)>,
}

impl EcPoint {
    pub(crate) fn loader(&self) -> &Rc<EvmLoader> {
        &self.loader
    }

    pub(crate) fn value(&self) -> Value<(U256, U256)> {
        self.value.clone()
    }

    pub(crate) fn ptr(&self) -> usize {
        match self.value {
            Value::Memory(ptr) => ptr,
            _ => unreachable!(),
        }
    }
}

impl Debug for EcPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EcPoint").field("value", &self.value).finish()
    }
}

impl PartialEq for EcPoint {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<C> LoadedEcPoint<C> for EcPoint
where
    C: CurveAffine,
    C::ScalarExt: PrimeField<Repr = [u8; 0x20]>,
{
    type Loader = Rc<EvmLoader>;

    fn loader(&self) -> &Rc<EvmLoader> {
        &self.loader
    }
}

/// Field element.
#[derive(Clone)]
pub struct Scalar {
    loader: Rc<EvmLoader>,
    value: Value<U256>,
}

impl Scalar {
    pub(crate) fn loader(&self) -> &Rc<EvmLoader> {
        &self.loader
    }

    pub(crate) fn value(&self) -> Value<U256> {
        self.value.clone()
    }

    pub(crate) fn is_const(&self) -> bool {
        matches!(self.value, Value::Constant(_))
    }

    pub(crate) fn ptr(&self) -> usize {
        match self.value {
            Value::Memory(ptr) => ptr,
            _ => *self.loader.cache.borrow().get(&self.value.identifier()).unwrap(),
        }
    }
}

impl Debug for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scalar").field("value", &self.value).finish()
    }
}

impl Add for Scalar {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        self.loader.add(&self, &rhs)
    }
}

impl Sub for Scalar {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self.loader.sub(&self, &rhs)
    }
}

impl Mul for Scalar {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        self.loader.mul(&self, &rhs)
    }
}

impl Neg for Scalar {
    type Output = Self;

    fn neg(self) -> Self {
        self.loader.neg(&self)
    }
}

impl<'a> Add<&'a Self> for Scalar {
    type Output = Self;

    fn add(self, rhs: &'a Self) -> Self {
        self.loader.add(&self, rhs)
    }
}

impl<'a> Sub<&'a Self> for Scalar {
    type Output = Self;

    fn sub(self, rhs: &'a Self) -> Self {
        self.loader.sub(&self, rhs)
    }
}

impl<'a> Mul<&'a Self> for Scalar {
    type Output = Self;

    fn mul(self, rhs: &'a Self) -> Self {
        self.loader.mul(&self, rhs)
    }
}

impl AddAssign for Scalar {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.loader.add(self, &rhs);
    }
}

impl SubAssign for Scalar {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.loader.sub(self, &rhs);
    }
}

impl MulAssign for Scalar {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.loader.mul(self, &rhs);
    }
}

impl<'a> AddAssign<&'a Self> for Scalar {
    fn add_assign(&mut self, rhs: &'a Self) {
        *self = self.loader.add(self, rhs);
    }
}

impl<'a> SubAssign<&'a Self> for Scalar {
    fn sub_assign(&mut self, rhs: &'a Self) {
        *self = self.loader.sub(self, rhs);
    }
}

impl<'a> MulAssign<&'a Self> for Scalar {
    fn mul_assign(&mut self, rhs: &'a Self) {
        *self = self.loader.mul(self, rhs);
    }
}

impl FieldOps for Scalar {
    fn invert(&self) -> Option<Scalar> {
        Some(self.loader.invert(self))
    }
}

impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<F: PrimeField<Repr = [u8; 0x20]>> LoadedScalar<F> for Scalar {
    type Loader = Rc<EvmLoader>;

    fn loader(&self) -> &Self::Loader {
        &self.loader
    }

    fn pow_var(&self, _exp: &Self, _exp_max_bits: usize) -> Self {
        todo!()
    }
}

impl<C> EcPointLoader<C> for Rc<EvmLoader>
where
    C: CurveAffine,
    C::Scalar: PrimeField<Repr = [u8; 0x20]>,
{
    type LoadedEcPoint = EcPoint;

    fn ec_point_load_const(&self, value: &C) -> EcPoint {
        let coordinates = value.coordinates().unwrap();
        let [x, y] = [coordinates.x(), coordinates.y()]
            .map(|coordinate| U256::try_from_le_slice(coordinate.to_repr().as_ref()).unwrap());
        self.ec_point(Value::Constant((x, y)))
    }

    fn ec_point_assert_eq(&self, _: &str, _: &EcPoint, _: &EcPoint) {
        unimplemented!()
    }

    fn multi_scalar_multiplication(
        pairs: &[(&<Self as ScalarLoader<C::Scalar>>::LoadedScalar, &EcPoint)],
    ) -> EcPoint {
        pairs
            .iter()
            .cloned()
            .map(|(scalar, ec_point)| match scalar.value {
                Value::Constant(constant) if U256::from(1) == constant => ec_point.clone(),
                _ => ec_point.loader.ec_point_scalar_mul(ec_point, scalar),
            })
            .reduce(|acc, ec_point| acc.loader.ec_point_add(&acc, &ec_point))
            .expect("pairs should not be empty")
    }
}

impl<F: PrimeField<Repr = [u8; 0x20]>> ScalarLoader<F> for Rc<EvmLoader> {
    type LoadedScalar = Scalar;

    fn load_const(&self, value: &F) -> Scalar {
        self.scalar(Value::Constant(fe_to_u256(*value)))
    }

    fn assert_eq(&self, _: &str, _: &Scalar, _: &Scalar) {
        unimplemented!()
    }

    fn sum_with_coeff_and_const(&self, values: &[(F, &Scalar)], constant: F) -> Scalar {
        if values.is_empty() {
            return self.load_const(&constant);
        }

        let push_addend = |(coeff, value): &(F, &Scalar)| {
            assert_ne!(*coeff, F::ZERO);
            match (*coeff == F::ONE, &value.value) {
                (true, _) => self.push(value),
                (false, Value::Constant(value)) => self.push(
                    &self.scalar(Value::Constant(fe_to_u256(*coeff * u256_to_fe::<F>(*value)))),
                ),
                (false, _) => {
                    let value = self.push(value);
                    let coeff = self.push(&self.scalar(Value::Constant(fe_to_u256(*coeff))));
                    format!("mulmod({value}, {coeff}, f_q)")
                }
            }
        };

        let mut values = values.iter();
        let initial_value = if constant == F::ZERO {
            push_addend(values.next().unwrap())
        } else {
            self.push(&self.scalar(Value::Constant(fe_to_u256(constant))))
        };

        let mut code = format!("let result := {initial_value}\n");
        for value in values {
            let v = push_addend(value);
            let addend = format!("result := addmod({v}, result, f_q)\n");
            code.push_str(addend.as_str());
        }

        let ptr = self.allocate(0x20);
        code.push_str(format!("mstore({ptr}, result)").as_str());
        self.code.borrow_mut().runtime_append(format!(
            "{{
            {code}
        }}"
        ));

        self.scalar(Value::Memory(ptr))
    }

    fn sum_products_with_coeff_and_const(
        &self,
        values: &[(F, &Scalar, &Scalar)],
        constant: F,
    ) -> Scalar {
        if values.is_empty() {
            return self.load_const(&constant);
        }

        let push_addend = |(coeff, lhs, rhs): &(F, &Scalar, &Scalar)| {
            assert_ne!(*coeff, F::ZERO);
            match (*coeff == F::ONE, &lhs.value, &rhs.value) {
                (_, Value::Constant(lhs), Value::Constant(rhs)) => {
                    self.push(&self.scalar(Value::Constant(fe_to_u256(
                        *coeff * u256_to_fe::<F>(*lhs) * u256_to_fe::<F>(*rhs),
                    ))))
                }
                (_, value @ Value::Memory(_), Value::Constant(constant))
                | (_, Value::Constant(constant), value @ Value::Memory(_)) => {
                    let v1 = self.push(&self.scalar(value.clone()));
                    let v2 =
                        self.push(&self.scalar(Value::Constant(fe_to_u256(
                            *coeff * u256_to_fe::<F>(*constant),
                        ))));
                    format!("mulmod({v1}, {v2}, f_q)")
                }
                (true, _, _) => {
                    let rhs = self.push(rhs);
                    let lhs = self.push(lhs);
                    format!("mulmod({rhs}, {lhs}, f_q)")
                }
                (false, _, _) => {
                    let rhs = self.push(rhs);
                    let lhs = self.push(lhs);
                    let value = self.push(&self.scalar(Value::Constant(fe_to_u256(*coeff))));
                    format!("mulmod({rhs}, mulmod({lhs}, {value}, f_q), f_q)")
                }
            }
        };

        let mut values = values.iter();
        let initial_value = if constant == F::ZERO {
            push_addend(values.next().unwrap())
        } else {
            self.push(&self.scalar(Value::Constant(fe_to_u256(constant))))
        };

        let mut code = format!("let result := {initial_value}\n");
        for value in values {
            let v = push_addend(value);
            let addend = format!("result := addmod({v}, result, f_q)\n");
            code.push_str(addend.as_str());
        }

        let ptr = self.allocate(0x20);
        code.push_str(format!("mstore({ptr}, result)").as_str());
        self.code.borrow_mut().runtime_append(format!(
            "{{
            {code}
        }}"
        ));

        self.scalar(Value::Memory(ptr))
    }

    // batch_invert algorithm
    // n := values.len() - 1
    // input : values[0], ..., values[n]
    // output : values[0]^{-1}, ..., values[n]^{-1}
    // 1. products[i] <- values[0] * ... * values[i], i = 1, ..., n
    // 2. inv <- (products[n])^{-1}
    // 3. v_n <- values[n]
    // 4. values[n] <- products[n - 1] * inv (values[n]^{-1})
    // 5. inv <- v_n * inv
    fn batch_invert<'a>(values: impl IntoIterator<Item = &'a mut Scalar>) {
        let values = values.into_iter().collect_vec();
        let loader = &values.first().unwrap().loader;
        let products = iter::once(values[0].clone())
            .chain(
                iter::repeat_with(|| loader.allocate(0x20))
                    .map(|ptr| loader.scalar(Value::Memory(ptr)))
                    .take(values.len() - 1),
            )
            .collect_vec();

        let initial_value = loader.push(products.first().unwrap());
        let mut code = format!("let prod := {initial_value}\n");
        for (value, product) in values.iter().zip(products.iter()).skip(1) {
            let v = loader.push(value);
            let ptr = product.ptr();
            code.push_str(
                format!(
                    "
                prod := mulmod({v}, prod, f_q)
                mstore({ptr:#x}, prod)
            "
                )
                .as_str(),
            );
        }
        loader.code.borrow_mut().runtime_append(format!(
            "{{
            {code}
        }}"
        ));

        let inv = loader.push(&loader.invert(products.last().unwrap()));

        let mut code = format!(
            "
            let inv := {inv}
            let v
        "
        );
        for (value, product) in
            values.iter().rev().zip(products.iter().rev().skip(1).map(Some).chain(iter::once(None)))
        {
            if let Some(product) = product {
                let val_ptr = value.ptr();
                let prod_ptr = product.ptr();
                let v = loader.push(value);
                code.push_str(
                    format!(
                        "
                    v := {v}
                    mstore({val_ptr}, mulmod(mload({prod_ptr:#x}), inv, f_q))
                    inv := mulmod(v, inv, f_q)
                "
                    )
                    .as_str(),
                );
            } else {
                let ptr = value.ptr();
                code.push_str(format!("mstore({ptr:#x}, inv)\n").as_str());
            }
        }
        loader.code.borrow_mut().runtime_append(format!(
            "{{
            {code}
        }}"
        ));
    }
}

impl<C> Loader<C> for Rc<EvmLoader>
where
    C: CurveAffine,
    C::Scalar: PrimeField<Repr = [u8; 0x20]>,
{
    #[cfg(test)]
    fn start_cost_metering(&self, identifier: &str) {
        self.start_gas_metering(identifier)
    }

    #[cfg(test)]
    fn end_cost_metering(&self) {
        self.end_gas_metering()
    }
}
