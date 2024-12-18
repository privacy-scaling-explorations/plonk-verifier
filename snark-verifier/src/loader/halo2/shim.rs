use crate::util::arithmetic::{CurveAffine, PrimeField};
use halo2_proofs::{
    circuit::{Cell, Value},
    plonk::Error,
};
use std::{fmt::Debug, ops::Deref};

/// Context for instructions.
pub trait Context: Debug {
    /// Enforce equality constraint on lhs and rhs.
    fn constrain_equal(&mut self, lhs: Cell, rhs: Cell) -> Result<(), Error>;

    /// Returns current region offset.
    fn offset(&self) -> usize;
}

/// Instructions to handle field element operations.
pub trait IntegerInstructions<'a, F: PrimeField>: Clone + Debug {
    /// Context.
    type Context: Context;
    /// Assigned cell.
    type AssignedCell: Clone + Debug;
    /// Assigned integer.
    type AssignedInteger: Clone + Debug;

    /// Assign an integer witness.
    fn assign_integer(
        &self,
        ctx: &mut Self::Context,
        integer: Value<F>,
    ) -> Result<Self::AssignedInteger, Error>;

    /// Assign an integer constant.
    fn assign_constant(
        &self,
        ctx: &mut Self::Context,
        integer: F,
    ) -> Result<Self::AssignedInteger, Error>;

    /// Sum integers with coefficients and constant.
    fn sum_with_coeff_and_const(
        &self,
        ctx: &mut Self::Context,
        values: &[(F, impl Deref<Target = Self::AssignedInteger>)],
        constant: F,
    ) -> Result<Self::AssignedInteger, Error>;

    /// Sum product of integers with coefficients and constant.
    fn sum_products_with_coeff_and_const(
        &self,
        ctx: &mut Self::Context,
        values: &[(
            F,
            impl Deref<Target = Self::AssignedInteger>,
            impl Deref<Target = Self::AssignedInteger>,
        )],
        constant: F,
    ) -> Result<Self::AssignedInteger, Error>;

    /// Returns `lhs - rhs`.
    fn sub(
        &self,
        ctx: &mut Self::Context,
        lhs: &Self::AssignedInteger,
        rhs: &Self::AssignedInteger,
    ) -> Result<Self::AssignedInteger, Error>;

    /// Returns `-value`.
    fn neg(
        &self,
        ctx: &mut Self::Context,
        value: &Self::AssignedInteger,
    ) -> Result<Self::AssignedInteger, Error>;

    /// Returns `1/value`.
    fn invert(
        &self,
        ctx: &mut Self::Context,
        value: &Self::AssignedInteger,
    ) -> Result<Self::AssignedInteger, Error>;

    /// Enforce `lhs` and `rhs` are equal.
    fn assert_equal(
        &self,
        ctx: &mut Self::Context,
        lhs: &Self::AssignedInteger,
        rhs: &Self::AssignedInteger,
    ) -> Result<(), Error>;
}

/// Instructions to handle elliptic curve point operations.
pub trait EccInstructions<'a, C: CurveAffine>: Clone + Debug {
    /// Context
    type Context: Context;
    /// [`IntegerInstructions`] to handle scalar field operation.
    type ScalarChip: IntegerInstructions<
        'a,
        C::Scalar,
        Context = Self::Context,
        AssignedCell = Self::AssignedCell,
        AssignedInteger = Self::AssignedScalar,
    >;
    /// Assigned cell.
    type AssignedCell: Clone + Debug;
    /// Assigned scalar field element.
    type AssignedScalar: Clone + Debug;
    /// Assigned elliptic curve point.
    type AssignedEcPoint: Clone + Debug;

    /// Returns reference of [`EccInstructions::ScalarChip`].
    fn scalar_chip(&self) -> &Self::ScalarChip;

    /// Assign an elliptic curve point constant.
    fn assign_constant(
        &self,
        ctx: &mut Self::Context,
        ec_point: C,
    ) -> Result<Self::AssignedEcPoint, Error>;

    /// Assign an elliptic curve point witness.
    fn assign_point(
        &self,
        ctx: &mut Self::Context,
        ec_point: Value<C>,
    ) -> Result<Self::AssignedEcPoint, Error>;

    /// Sum elliptic curve points and constant.
    fn sum_with_const(
        &self,
        ctx: &mut Self::Context,
        values: &[impl Deref<Target = Self::AssignedEcPoint>],
        constant: C,
    ) -> Result<Self::AssignedEcPoint, Error>;

    /// Perform fixed base multi-scalar multiplication.
    fn fixed_base_msm(
        &mut self,
        ctx: &mut Self::Context,
        pairs: &[(impl Deref<Target = Self::AssignedScalar>, C)],
    ) -> Result<Self::AssignedEcPoint, Error>;

    /// Perform variable base multi-scalar multiplication.
    fn variable_base_msm(
        &mut self,
        ctx: &mut Self::Context,
        pairs: &[(
            impl Deref<Target = Self::AssignedScalar>,
            impl Deref<Target = Self::AssignedEcPoint>,
        )],
    ) -> Result<Self::AssignedEcPoint, Error>;

    /// Enforce `lhs` and `rhs` are equal.
    fn assert_equal(
        &self,
        ctx: &mut Self::Context,
        lhs: &Self::AssignedEcPoint,
        rhs: &Self::AssignedEcPoint,
    ) -> Result<(), Error>;
}

mod halo2_wrong {
    use crate::{
        loader::halo2::{Context, EccInstructions, IntegerInstructions},
        util::{
            arithmetic::{CurveAffine, Group, PrimeField},
            Itertools,
        },
    };
    use halo2_proofs::{
        circuit::{AssignedCell, Cell, Value},
        plonk::Error,
    };
    use halo2_wrong_ecc::{
        integer::rns::Common,
        maingate::{
            CombinationOption, CombinationOptionCommon, MainGate, MainGateInstructions, RegionCtx,
            Term,
        },
        AssignedPoint, BaseFieldEccChip,
    };
    use rand::rngs::OsRng;
    use std::{iter, ops::Deref};

    impl<'a, F: PrimeField> Context for RegionCtx<'a, F> {
        fn constrain_equal(&mut self, lhs: Cell, rhs: Cell) -> Result<(), Error> {
            self.constrain_equal(lhs, rhs)
        }

        fn offset(&self) -> usize {
            self.offset()
        }
    }

    impl<'a, F: PrimeField> IntegerInstructions<'a, F> for MainGate<F> {
        type Context = RegionCtx<'a, F>;
        type AssignedCell = AssignedCell<F, F>;
        type AssignedInteger = AssignedCell<F, F>;

        fn assign_integer(
            &self,
            ctx: &mut Self::Context,
            integer: Value<F>,
        ) -> Result<Self::AssignedInteger, Error> {
            self.assign_value(ctx, integer)
        }

        fn assign_constant(
            &self,
            ctx: &mut Self::Context,
            integer: F,
        ) -> Result<Self::AssignedInteger, Error> {
            MainGateInstructions::assign_constant(self, ctx, integer)
        }

        fn sum_with_coeff_and_const(
            &self,
            ctx: &mut Self::Context,
            values: &[(F, impl Deref<Target = Self::AssignedInteger>)],
            constant: F,
        ) -> Result<Self::AssignedInteger, Error> {
            self.compose(
                ctx,
                &values
                    .iter()
                    .map(|(coeff, assigned)| Term::Assigned(assigned, *coeff))
                    .collect_vec(),
                constant,
            )
        }

        fn sum_products_with_coeff_and_const(
            &self,
            ctx: &mut Self::Context,
            values: &[(
                F,
                impl Deref<Target = Self::AssignedInteger>,
                impl Deref<Target = Self::AssignedInteger>,
            )],
            constant: F,
        ) -> Result<Self::AssignedInteger, Error> {
            match values.len() {
                0 => MainGateInstructions::assign_constant(self, ctx, constant),
                1 => {
                    let (scalar, lhs, rhs) = &values[0];
                    let output = lhs
                        .value()
                        .zip(rhs.value())
                        .map(|(lhs, rhs)| *scalar * lhs * rhs + constant);

                    Ok(self
                        .apply(
                            ctx,
                            [
                                Term::Zero,
                                Term::Zero,
                                Term::assigned_to_mul(lhs),
                                Term::assigned_to_mul(rhs),
                                Term::unassigned_to_sub(output),
                            ],
                            constant,
                            CombinationOption::OneLinerDoubleMul(*scalar),
                        )?
                        .swap_remove(4))
                }
                _ => {
                    let (scalar, lhs, rhs) = &values[0];
                    self.apply(
                        ctx,
                        [Term::assigned_to_mul(lhs), Term::assigned_to_mul(rhs)],
                        constant,
                        CombinationOptionCommon::CombineToNextScaleMul(-F::ONE, *scalar).into(),
                    )?;
                    let acc =
                        Value::known(*scalar) * lhs.value() * rhs.value() + Value::known(constant);
                    let output =
                        values
                            .iter()
                            .skip(1)
                            .try_fold(acc, |acc, (scalar, lhs, rhs)| {
                                self.apply(
                                    ctx,
                                    [
                                        Term::assigned_to_mul(lhs),
                                        Term::assigned_to_mul(rhs),
                                        Term::Zero,
                                        Term::Zero,
                                        Term::Unassigned(acc, F::ONE),
                                    ],
                                    F::ZERO,
                                    CombinationOptionCommon::CombineToNextScaleMul(
                                        -F::ONE,
                                        *scalar,
                                    )
                                    .into(),
                                )?;
                                Ok::<_, Error>(
                                    acc + Value::known(*scalar) * lhs.value() * rhs.value(),
                                )
                            })?;
                    self.apply(
                        ctx,
                        [
                            Term::Zero,
                            Term::Zero,
                            Term::Zero,
                            Term::Zero,
                            Term::Unassigned(output, F::ZERO),
                        ],
                        F::ZERO,
                        CombinationOptionCommon::OneLinerAdd.into(),
                    )
                    .map(|mut outputs| outputs.swap_remove(4))
                }
            }
        }

        fn sub(
            &self,
            ctx: &mut Self::Context,
            lhs: &Self::AssignedInteger,
            rhs: &Self::AssignedInteger,
        ) -> Result<Self::AssignedInteger, Error> {
            MainGateInstructions::sub(self, ctx, lhs, rhs)
        }

        fn neg(
            &self,
            ctx: &mut Self::Context,
            value: &Self::AssignedInteger,
        ) -> Result<Self::AssignedInteger, Error> {
            MainGateInstructions::neg_with_constant(self, ctx, value, F::ZERO)
        }

        fn invert(
            &self,
            ctx: &mut Self::Context,
            value: &Self::AssignedInteger,
        ) -> Result<Self::AssignedInteger, Error> {
            MainGateInstructions::invert_unsafe(self, ctx, value)
        }

        fn assert_equal(
            &self,
            ctx: &mut Self::Context,
            lhs: &Self::AssignedInteger,
            rhs: &Self::AssignedInteger,
        ) -> Result<(), Error> {
            let mut eq = true;
            lhs.value().zip(rhs.value()).map(|(lhs, rhs)| {
                eq &= lhs == rhs;
            });
            MainGateInstructions::assert_equal(self, ctx, lhs, rhs)
                .and(eq.then_some(()).ok_or(Error::Synthesis))
        }
    }

    impl<'a, C: CurveAffine, const LIMBS: usize, const BITS: usize> EccInstructions<'a, C>
        for BaseFieldEccChip<C, LIMBS, BITS>
    {
        type Context = RegionCtx<'a, C::Scalar>;
        type ScalarChip = MainGate<C::Scalar>;
        type AssignedCell = AssignedCell<C::Scalar, C::Scalar>;
        type AssignedScalar = AssignedCell<C::Scalar, C::Scalar>;
        type AssignedEcPoint = AssignedPoint<C::Base, C::Scalar, LIMBS, BITS>;

        fn scalar_chip(&self) -> &Self::ScalarChip {
            self.main_gate()
        }

        fn assign_constant(
            &self,
            ctx: &mut Self::Context,
            ec_point: C,
        ) -> Result<Self::AssignedEcPoint, Error> {
            self.assign_constant(ctx, ec_point)
        }

        fn assign_point(
            &self,
            ctx: &mut Self::Context,
            ec_point: Value<C>,
        ) -> Result<Self::AssignedEcPoint, Error> {
            self.assign_point(ctx, ec_point)
        }

        fn sum_with_const(
            &self,
            ctx: &mut Self::Context,
            values: &[impl Deref<Target = Self::AssignedEcPoint>],
            constant: C,
        ) -> Result<Self::AssignedEcPoint, Error> {
            if values.is_empty() {
                return self.assign_constant(ctx, constant);
            }

            let constant = (!bool::from(constant.is_identity()))
                .then(|| self.assign_constant(ctx, constant))
                .transpose()?;
            let output = iter::empty()
                .chain(constant)
                .chain(values.iter().map(|value| value.deref().clone()))
                .map(Ok)
                .reduce(|acc, ec_point| self.add(ctx, &acc?, &ec_point?))
                .unwrap()?;
            self.normalize(ctx, &output)
        }

        fn fixed_base_msm(
            &mut self,
            ctx: &mut Self::Context,
            pairs: &[(impl Deref<Target = Self::AssignedScalar>, C)],
        ) -> Result<Self::AssignedEcPoint, Error> {
            assert!(!pairs.is_empty());

            // FIXME: Implement fixed base MSM in halo2_wrong
            let pairs = pairs
                .iter()
                .filter(|(_, base)| !bool::from(base.is_identity()))
                .map(|(scalar, base)| {
                    Ok::<_, Error>((scalar.deref().clone(), self.assign_constant(ctx, *base)?))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let pairs = pairs.iter().map(|tup| (&tup.0, &tup.1)).collect_vec();
            self.variable_base_msm(ctx, &pairs)
        }

        fn variable_base_msm(
            &mut self,
            ctx: &mut Self::Context,
            pairs: &[(
                impl Deref<Target = Self::AssignedScalar>,
                impl Deref<Target = Self::AssignedEcPoint>,
            )],
        ) -> Result<Self::AssignedEcPoint, Error> {
            assert!(!pairs.is_empty());

            const WINDOW_SIZE: usize = 3;
            let pairs = pairs
                .iter()
                .map(|(scalar, base)| (base.deref().clone(), scalar.deref().clone()))
                .collect_vec();
            let output = match self.mul_batch_1d_horizontal(ctx, pairs.clone(), WINDOW_SIZE) {
                Err(_) => {
                    if self.assign_aux(ctx, WINDOW_SIZE, pairs.len()).is_err() {
                        let aux_generator = Value::known(C::Curve::random(OsRng).into());
                        self.assign_aux_generator(ctx, aux_generator)?;
                        self.assign_aux(ctx, WINDOW_SIZE, pairs.len())?;
                    }
                    self.mul_batch_1d_horizontal(ctx, pairs, WINDOW_SIZE)
                }
                result => result,
            }?;
            self.normalize(ctx, &output)
        }

        fn assert_equal(
            &self,
            ctx: &mut Self::Context,
            lhs: &Self::AssignedEcPoint,
            rhs: &Self::AssignedEcPoint,
        ) -> Result<(), Error> {
            let mut eq = true;
            [(lhs.x(), rhs.x()), (lhs.y(), rhs.y())].map(|(lhs, rhs)| {
                lhs.integer().zip(rhs.integer()).map(|(lhs, rhs)| {
                    eq &= lhs.value() == rhs.value();
                });
            });
            self.assert_equal(ctx, lhs, rhs)
                .and(eq.then_some(()).ok_or(Error::Synthesis))
        }
    }
}
