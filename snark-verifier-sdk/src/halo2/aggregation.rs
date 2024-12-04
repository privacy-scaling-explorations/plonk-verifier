use super::PlonkSuccinctVerifier;
use crate::{BITS, LIMBS};
use getset::Getters;
use halo2_base::{
    gates::{
        circuit::{
            builder::BaseCircuitBuilder, BaseCircuitParams, BaseConfig, CircuitBuilderStage,
        },
        flex_gate::{threads::SinglePhaseCoreManager, MultiPhaseThreadBreakPoints},
        RangeChip,
    },
    halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner},
        halo2curves::bn256::{Bn256, Fr, G1Affine},
        plonk::{self, Circuit, ConstraintSystem, Selector},
        poly::{commitment::ParamsProver, kzg::commitment::ParamsKZG},
    },
    utils::ScalarField,
    AssignedValue,
};
use itertools::Itertools;
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
// #[cfg(debug_assertions)]
// use snark_verifier::util::arithmetic::fe_to_limbs;
use snark_verifier::{
    loader::{
        self,
        halo2::halo2_ecc::{self, bigint::ProperCrtUint, bn254::FpChip},
        native::NativeLoader,
    },
    pcs::{
        kzg::{KzgAccumulator, KzgAsProvingKey, KzgAsVerifyingKey, KzgSuccinctVerifyingKey},
        AccumulationScheme, AccumulationSchemeProver, PolynomialCommitmentScheme,
    },
    system::halo2::transcript::halo2::TranscriptObject,
    verifier::SnarkVerifier,
};
use std::{fs::File, mem, path::Path, rc::Rc};

use super::{CircuitExt, PoseidonTranscript, Snark, POSEIDON_SPEC};

pub type Svk = KzgSuccinctVerifyingKey<G1Affine>;
pub type BaseFieldEccChip<'chip> = halo2_ecc::ecc::BaseFieldEccChip<'chip, G1Affine>;
pub type Halo2Loader<'chip> = loader::halo2::Halo2Loader<G1Affine, BaseFieldEccChip<'chip>>;

#[derive(Clone, Debug)]
pub struct PreprocessedAndDomainAsWitness {
    // this is basically the vkey
    pub preprocessed: Vec<AssignedValue<Fr>>,
    pub k: AssignedValue<Fr>,
}

#[derive(Clone, Debug)]
pub struct SnarkAggregationWitness<'a> {
    /// The (flattened) public instances from previous snarks that were aggregated, now collected as PRIVATE assigned values.
    /// * If previous snark was from aggregation circuit, the previous instances will still contain the old KZG accumulator.
    ///
    /// The user can optionally append these private witnesses to `inner.assigned_instances` to expose them.
    pub previous_instances: Vec<Vec<AssignedValue<Fr>>>,
    pub accumulator: KzgAccumulator<G1Affine, Rc<Halo2Loader<'a>>>,
    /// This returns the assigned `preprocessed` and `transcript_initial_state` values as a vector of assigned values, one for each aggregated snark.
    /// These can then be exposed as public instances.
    pub preprocessed: Vec<PreprocessedAndDomainAsWitness>,
    /// The proof transcript, as loaded scalars and elliptic curve points, for each SNARK that was aggregated.
    pub proof_transcripts: Vec<Vec<TranscriptObject<G1Affine, Rc<Halo2Loader<'a>>>>>,
}

/// Different possible stages of universality the aggregation circuit can support
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default)]
pub enum VerifierUniversality {
    /// Default: verifier is specific to a single circuit
    #[default]
    None,
    /// Preprocessed digest (commitments to fixed columns) is loaded as witness
    PreprocessedAsWitness,
    /// Preprocessed as witness and log_2(number of rows in the circuit) = k loaded as witness
    Full,
}

impl VerifierUniversality {
    pub fn preprocessed_as_witness(&self) -> bool {
        self != &VerifierUniversality::None
    }

    pub fn k_as_witness(&self) -> bool {
        self == &VerifierUniversality::Full
    }
}

#[allow(clippy::type_complexity)]
/// Core function used in `synthesize` to aggregate multiple `snarks`.
///  
/// Returns the assigned instances of previous snarks and the new final pair that needs to be verified in a pairing check.
/// For each previous snark, we concatenate all instances into a single vector. We return a vector of vectors,
/// one vector per snark, for convenience.
///
/// - `preprocessed_as_witness`: flag for whether preprocessed digest (i.e., verifying key) should be loaded as witness (if false, loaded as constant)
///     - If `preprocessed_as_witness` is true, number of circuit rows `domain.n` is also loaded as a witness
///
/// # Assumptions
/// * `snarks` is not empty
pub fn aggregate<'a, AS>(
    svk: &Svk,
    loader: &Rc<Halo2Loader<'a>>,
    snarks: &[Snark],
    as_proof: &[u8],
    universality: VerifierUniversality,
) -> SnarkAggregationWitness<'a>
where
    AS: PolynomialCommitmentScheme<
            G1Affine,
            Rc<Halo2Loader<'a>>,
            VerifyingKey = Svk,
            Output = KzgAccumulator<G1Affine, Rc<Halo2Loader<'a>>>,
        > + AccumulationScheme<
            G1Affine,
            Rc<Halo2Loader<'a>>,
            Accumulator = KzgAccumulator<G1Affine, Rc<Halo2Loader<'a>>>,
            VerifyingKey = KzgAsVerifyingKey,
        >,
{
    assert!(!snarks.is_empty(), "trying to aggregate 0 snarks");
    let assign_instances = |instances: &[Vec<Fr>]| {
        instances
            .iter()
            .map(|instances| {
                instances.iter().map(|instance| loader.assign_scalar(*instance)).collect_vec()
            })
            .collect_vec()
    };

    let mut previous_instances = Vec::with_capacity(snarks.len());
    let mut preprocessed_witnesses = Vec::with_capacity(snarks.len());
    // to avoid re-loading the spec each time, we create one transcript and clear the stream
    let mut transcript = PoseidonTranscript::<Rc<Halo2Loader<'a>>, &[u8]>::from_spec(
        loader,
        &[],
        POSEIDON_SPEC.clone(),
    );

    let preprocessed_as_witness = universality.preprocessed_as_witness();
    let (proof_transcripts, accumulators): (Vec<_>, Vec<_>) = snarks
        .iter()
        .map(|snark: &Snark| {
            let protocol = if preprocessed_as_witness {
                // always load `domain.n` as witness if vkey is witness
                snark.protocol.loaded_preprocessed_as_witness(loader, universality.k_as_witness())
            } else {
                snark.protocol.loaded(loader)
            };
            let preprocessed = protocol
                .preprocessed
                .iter()
                .flat_map(|preprocessed| {
                    let assigned = preprocessed.assigned();
                    [assigned.x(), assigned.y()]
                        .into_iter()
                        .flat_map(|coordinate| coordinate.limbs().to_vec())
                        .collect_vec()
                })
                .chain(
                    protocol.transcript_initial_state.clone().map(|scalar| scalar.into_assigned()),
                )
                .collect_vec();
            // Store `k` as witness. If `k` was fixed, assign it as a constant.
            let k = protocol
                .domain_as_witness
                .as_ref()
                .map(|domain| domain.k.clone().into_assigned())
                .unwrap_or_else(|| {
                    loader.ctx_mut().main().load_constant(Fr::from(protocol.domain.k as u64))
                });
            let preprocessed_and_k = PreprocessedAndDomainAsWitness { preprocessed, k };
            preprocessed_witnesses.push(preprocessed_and_k);

            let instances = assign_instances(&snark.instances);

            // read the transcript and perform Fiat-Shamir
            // run through verification computation and produce the final pair `succinct`
            transcript.new_stream(snark.proof());
            let proof = PlonkSuccinctVerifier::<AS>::read_proof(
                svk,
                &protocol,
                &instances,
                &mut transcript,
            )
            .unwrap();
            let accumulator =
                PlonkSuccinctVerifier::<AS>::verify(svk, &protocol, &instances, &proof).unwrap();

            previous_instances.push(
                instances.into_iter().flatten().map(|scalar| scalar.into_assigned()).collect(),
            );
            let proof_transcript = transcript.loaded_stream.clone();
            debug_assert_eq!(
                snark.proof().len(),
                proof_transcript
                    .iter()
                    .map(|t| match t {
                        TranscriptObject::Scalar(_) => 32,
                        TranscriptObject::EcPoint(_) => 32,
                    })
                    .sum::<usize>()
            );
            (proof_transcript, accumulator)
        })
        .unzip();
    let mut accumulators = accumulators.into_iter().flatten().collect_vec();

    let accumulator = if accumulators.len() > 1 {
        transcript.new_stream(as_proof);
        let proof = <AS as AccumulationScheme<_, _>>::read_proof(
            &Default::default(),
            &accumulators,
            &mut transcript,
        )
        .unwrap();
        <AS as AccumulationScheme<_, _>>::verify(&Default::default(), &accumulators, &proof)
            .unwrap()
    } else {
        accumulators.pop().unwrap()
    };

    SnarkAggregationWitness {
        previous_instances,
        accumulator,
        preprocessed: preprocessed_witnesses,
        proof_transcripts,
    }
}

/// Same as `FlexGateConfigParams` except we assume a single Phase and default 'Vertical' strategy.
/// Also adds `lookup_bits` field.
#[derive(Clone, Copy, Default, Debug, Serialize, Deserialize)]
pub struct AggregationConfigParams {
    pub degree: u32,
    pub num_advice: usize,
    pub num_lookup_advice: usize,
    pub num_fixed: usize,
    pub lookup_bits: usize,
}

impl AggregationConfigParams {
    pub fn from_path(path: impl AsRef<Path>) -> Self {
        serde_json::from_reader(File::open(path).expect("Aggregation config path does not exist"))
            .unwrap()
    }
}

impl From<AggregationConfigParams> for BaseCircuitParams {
    fn from(params: AggregationConfigParams) -> Self {
        BaseCircuitParams {
            k: params.degree as usize,
            num_advice_per_phase: vec![params.num_advice],
            num_lookup_advice_per_phase: vec![params.num_lookup_advice],
            num_fixed: params.num_fixed,
            lookup_bits: Some(params.lookup_bits),
            num_instance_columns: 1,
        }
    }
}

impl TryFrom<&BaseCircuitParams> for AggregationConfigParams {
    type Error = &'static str;

    fn try_from(params: &BaseCircuitParams) -> Result<Self, Self::Error> {
        if params.num_advice_per_phase.iter().skip(1).any(|&n| n != 0) {
            return Err("AggregationConfigParams only supports 1 phase");
        }
        if params.num_lookup_advice_per_phase.iter().skip(1).any(|&n| n != 0) {
            return Err("AggregationConfigParams only supports 1 phase");
        }
        if params.lookup_bits.is_none() {
            return Err("AggregationConfigParams requires lookup_bits");
        }
        if params.num_instance_columns != 1 {
            return Err("AggregationConfigParams only supports 1 instance column");
        }
        Ok(Self {
            degree: params.k as u32,
            num_advice: params.num_advice_per_phase[0],
            num_lookup_advice: params.num_lookup_advice_per_phase[0],
            num_fixed: params.num_fixed,
            lookup_bits: params.lookup_bits.unwrap(),
        })
    }
}

impl TryFrom<BaseCircuitParams> for AggregationConfigParams {
    type Error = &'static str;

    fn try_from(value: BaseCircuitParams) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

#[derive(Clone, Debug, Getters)]
pub struct AggregationCircuit {
    /// Circuit builder consisting of virtual region managers
    pub builder: BaseCircuitBuilder<Fr>,
    /// The (flattened) public instances from previous snarks that were aggregated, now collected as PRIVATE assigned values.
    /// * If previous snark was from aggregation circuit, the previous instances will still contain the old KZG accumulator.
    ///
    /// The user can optionally append these private witnesses to `inner.assigned_instances` to expose them.
    #[getset(get = "pub")]
    previous_instances: Vec<Vec<AssignedValue<Fr>>>,
    /// This returns the assigned `preprocessed_digest` (vkey), optional `transcript_initial_state`, `domain.n` (optional), and `omega` (optional) values as a vector of assigned values, one for each aggregated snark.
    /// These can then be exposed as public instances.
    #[getset(get = "pub")]
    preprocessed: Vec<PreprocessedAndDomainAsWitness>,
    // accumulation scheme proof, no longer used
    // pub as_proof: Vec<u8>,
}

// trait just so we can have a generic that is either SHPLONK or GWC
pub trait Halo2KzgAccumulationScheme<'a> = PolynomialCommitmentScheme<
        G1Affine,
        Rc<Halo2Loader<'a>>,
        VerifyingKey = Svk,
        Output = KzgAccumulator<G1Affine, Rc<Halo2Loader<'a>>>,
    > + AccumulationScheme<
        G1Affine,
        Rc<Halo2Loader<'a>>,
        Accumulator = KzgAccumulator<G1Affine, Rc<Halo2Loader<'a>>>,
        VerifyingKey = KzgAsVerifyingKey,
    > + PolynomialCommitmentScheme<
        G1Affine,
        NativeLoader,
        VerifyingKey = Svk,
        Output = KzgAccumulator<G1Affine, NativeLoader>,
    > + AccumulationScheme<
        G1Affine,
        NativeLoader,
        Accumulator = KzgAccumulator<G1Affine, NativeLoader>,
        VerifyingKey = KzgAsVerifyingKey,
    > + AccumulationSchemeProver<G1Affine, ProvingKey = KzgAsProvingKey<G1Affine>>;

/// **Private** witnesses that form the output of [aggregate_snarks].
/// Same as [SnarkAggregationWitness] except that we flatten `accumulator` into a vector of field elements.
#[derive(Clone, Debug)]
pub struct SnarkAggregationOutput {
    pub previous_instances: Vec<Vec<AssignedValue<Fr>>>,
    pub accumulator: Vec<AssignedValue<Fr>>,
    /// This returns the assigned `preprocessed` and `transcript_initial_state` values as a vector of assigned values, one for each aggregated snark.
    /// These can then be exposed as public instances.
    pub preprocessed: Vec<PreprocessedAndDomainAsWitness>,
    /// The proof transcript, as loaded scalars and elliptic curve points, for each SNARK that was aggregated.
    pub proof_transcripts: Vec<Vec<AssignedTranscriptObject>>,
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum AssignedTranscriptObject {
    Scalar(AssignedValue<Fr>),
    EcPoint(halo2_ecc::ecc::EcPoint<Fr, ProperCrtUint<Fr>>),
}

/// Given snarks, this populates the circuit builder with the virtual cells and constraints necessary to verify all the snarks.
///
/// ## Notes
/// - This function does _not_ expose any public instances.
/// - `svk` is the generator of the KZG trusted setup, usually gotten via `params.get_g()[0]`
/// (avoids having to pass `params` into function just to get generator)
///
/// ## Universality
/// - If `universality` is not `None`, then the verifying keys of each snark in `snarks` is loaded as a witness in the circuit.
/// - Moreover, if `universality` is `Full`, then the number of rows `n` of each snark in `snarks` is also loaded as a witness. In this case the generator `omega` of the order `n` multiplicative subgroup of `F` is also loaded as a witness.
/// - By default, these witnesses are _private_ and returned in `self.preprocessed_digests
/// - The user can optionally modify the circuit after calling this function to add more instances to `assigned_instances` to expose.
///
/// ## Warning
/// Will fail silently if `snarks` were created using a different multi-open scheme than `AS`
/// where `AS` can be either [`crate::SHPLONK`] or [`crate::GWC`] (for original PLONK multi-open scheme)
///
/// ## Assumptions
/// - `pool` and `range` reference the same `SharedCopyConstraintManager`.
pub fn aggregate_snarks<AS>(
    pool: &mut SinglePhaseCoreManager<Fr>,
    range: &RangeChip<Fr>,
    svk: Svk, // gotten by params.get_g()[0].into()
    snarks: impl IntoIterator<Item = Snark>,
    universality: VerifierUniversality,
) -> SnarkAggregationOutput
where
    AS: for<'a> Halo2KzgAccumulationScheme<'a>,
{
    let snarks = snarks.into_iter().collect_vec();

    let mut transcript_read =
        PoseidonTranscript::<NativeLoader, &[u8]>::from_spec(&[], POSEIDON_SPEC.clone());
    // TODO: the snarks can probably store these accumulators
    let accumulators = snarks
        .iter()
        .flat_map(|snark| {
            transcript_read.new_stream(snark.proof());
            let proof = PlonkSuccinctVerifier::<AS>::read_proof(
                &svk,
                &snark.protocol,
                &snark.instances,
                &mut transcript_read,
            )
            .unwrap();
            PlonkSuccinctVerifier::<AS>::verify(&svk, &snark.protocol, &snark.instances, &proof)
                .unwrap()
        })
        .collect_vec();

    let (_accumulator, as_proof) = {
        let mut transcript_write =
            PoseidonTranscript::<NativeLoader, Vec<u8>>::from_spec(vec![], POSEIDON_SPEC.clone());
        let rng = StdRng::from_entropy();
        let accumulator =
            AS::create_proof(&Default::default(), &accumulators, &mut transcript_write, rng)
                .unwrap();
        (accumulator, transcript_write.finalize())
    };

    // create halo2loader
    let fp_chip = FpChip::<Fr>::new(range, BITS, LIMBS);
    let ecc_chip = BaseFieldEccChip::new(&fp_chip);
    // `pool` needs to be owned by loader.
    // We put it back later (below), so it should have same effect as just mutating `pool`.
    let tmp_pool = mem::take(pool);
    // range_chip has shared reference to LookupAnyManager, with shared CopyConstraintManager
    // pool has shared reference to CopyConstraintManager
    let loader = Halo2Loader::new(ecc_chip, tmp_pool);

    // run witness and copy constraint generation
    let SnarkAggregationWitness {
        previous_instances,
        accumulator,
        preprocessed,
        proof_transcripts,
    } = aggregate::<AS>(&svk, &loader, &snarks, as_proof.as_slice(), universality);
    let lhs = accumulator.lhs.assigned();
    let rhs = accumulator.rhs.assigned();
    let accumulator = lhs
        .x()
        .limbs()
        .iter()
        .chain(lhs.y().limbs().iter())
        .chain(rhs.x().limbs().iter())
        .chain(rhs.y().limbs().iter())
        .copied()
        .collect_vec();
    let proof_transcripts = proof_transcripts
        .into_iter()
        .map(|transcript| {
            transcript
                .into_iter()
                .map(|obj| match obj {
                    TranscriptObject::Scalar(scalar) => {
                        AssignedTranscriptObject::Scalar(scalar.into_assigned())
                    }
                    TranscriptObject::EcPoint(point) => {
                        AssignedTranscriptObject::EcPoint(point.into_assigned())
                    }
                })
                .collect()
        })
        .collect();

    // #[cfg(debug_assertions)]
    // {
    //     let KzgAccumulator { lhs, rhs } = _accumulator;
    //     let instances =
    //         [lhs.x, lhs.y, rhs.x, rhs.y].map(fe_to_limbs::<_, Fr, LIMBS, BITS>).concat();
    //     for (lhs, rhs) in instances.iter().zip(accumulator.iter()) {
    //         assert_eq!(lhs, rhs.value());
    //     }
    // }
    // put back `pool` into `builder`
    *pool = loader.take_ctx();
    SnarkAggregationOutput { previous_instances, accumulator, preprocessed, proof_transcripts }
}

impl AggregationCircuit {
    /// Given snarks, this creates `BaseCircuitBuilder` and populates the circuit builder with the virtual cells and constraints necessary to verify all the snarks.
    ///
    /// By default, the returned circuit has public instances equal to the limbs of the pair of elliptic curve points, referred to as the `accumulator`, that need to be verified in a final pairing check.
    ///
    /// # Universality
    /// - If `universality` is not `None`, then the verifying keys of each snark in `snarks` is loaded as a witness in the circuit.
    /// - Moreover, if `universality` is `Full`, then the number of rows `n` of each snark in `snarks` is also loaded as a witness. In this case the generator `omega` of the order `n` multiplicative subgroup of `F` is also loaded as a witness.
    /// - By default, these witnesses are _private_ and returned in `self.preprocessed_digests
    /// - The user can optionally modify the circuit after calling this function to add more instances to `assigned_instances` to expose.
    ///
    /// # Warning
    /// Will fail silently if `snarks` were created using a different multi-open scheme than `AS`
    /// where `AS` can be either [`crate::SHPLONK`] or [`crate::GWC`] (for original PLONK multi-open scheme)
    pub fn new<AS>(
        stage: CircuitBuilderStage,
        config_params: AggregationConfigParams,
        params: &ParamsKZG<Bn256>,
        snarks: impl IntoIterator<Item = Snark>,
        universality: VerifierUniversality,
    ) -> Self
    where
        AS: for<'a> Halo2KzgAccumulationScheme<'a>,
    {
        let svk: Svk = params.get_g()[0].into();
        let mut builder = BaseCircuitBuilder::from_stage(stage).use_params(config_params.into());
        let range = builder.range_chip();
        let SnarkAggregationOutput { previous_instances, accumulator, preprocessed, .. } =
            aggregate_snarks::<AS>(builder.pool(0), &range, svk, snarks, universality);
        assert_eq!(
            builder.assigned_instances.len(),
            1,
            "AggregationCircuit must have exactly 1 instance column"
        );
        // expose accumulator as public instances
        builder.assigned_instances[0] = accumulator;
        Self { builder, previous_instances, preprocessed }
    }

    /// Re-expose the previous public instances of aggregated snarks again.
    /// If `hash_prev_accumulator` is true, then we assume all aggregated snarks were themselves
    /// aggregation snarks, and we exclude the old accumulators from the public input.
    pub fn expose_previous_instances(&mut self, has_prev_accumulator: bool) {
        let start = (has_prev_accumulator as usize) * 4 * LIMBS;
        for prev in self.previous_instances.iter() {
            self.builder.assigned_instances[0].extend_from_slice(&prev[start..]);
        }
    }

    /// The log_2 size of the lookup table
    pub fn lookup_bits(&self) -> usize {
        self.builder.config_params.lookup_bits.unwrap()
    }

    /// Set config params
    pub fn set_params(&mut self, params: AggregationConfigParams) {
        self.builder.set_params(params.into());
    }

    /// Returns new with config params
    pub fn use_params(mut self, params: AggregationConfigParams) -> Self {
        self.set_params(params);
        self
    }

    /// The break points of the circuit.
    pub fn break_points(&self) -> MultiPhaseThreadBreakPoints {
        self.builder.break_points()
    }

    /// Sets the break points of the circuit.
    pub fn set_break_points(&mut self, break_points: MultiPhaseThreadBreakPoints) {
        self.builder.set_break_points(break_points);
    }

    /// Returns new with break points
    pub fn use_break_points(mut self, break_points: MultiPhaseThreadBreakPoints) -> Self {
        self.set_break_points(break_points);
        self
    }

    /// Auto-configure the circuit and change the circuit's internal configuration parameters.
    pub fn calculate_params(&mut self, minimum_rows: Option<usize>) -> AggregationConfigParams {
        self.builder.calculate_params(minimum_rows).try_into().unwrap()
    }
}

impl<F: ScalarField> CircuitExt<F> for BaseCircuitBuilder<F> {
    fn num_instance(&self) -> Vec<usize> {
        self.assigned_instances.iter().map(|instances| instances.len()).collect()
    }

    fn instances(&self) -> Vec<Vec<F>> {
        self.assigned_instances
            .iter()
            .map(|instances| instances.iter().map(|v| *v.value()).collect())
            .collect()
    }

    fn selectors(config: &Self::Config) -> Vec<Selector> {
        config.gate().basic_gates[0].iter().map(|gate| gate.q_enable).collect()
    }
}

impl Circuit<Fr> for AggregationCircuit {
    type Config = BaseConfig<Fr>;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = AggregationConfigParams;

    fn params(&self) -> Self::Params {
        (&self.builder.config_params).try_into().unwrap()
    }

    fn without_witnesses(&self) -> Self {
        unimplemented!()
    }

    fn configure_with_params(
        meta: &mut ConstraintSystem<Fr>,
        params: Self::Params,
    ) -> Self::Config {
        BaseCircuitBuilder::configure_with_params(meta, params.into())
    }

    fn configure(_: &mut ConstraintSystem<Fr>) -> Self::Config {
        unreachable!()
    }

    fn synthesize(
        &self,
        config: Self::Config,
        layouter: impl Layouter<Fr>,
    ) -> Result<(), plonk::Error> {
        self.builder.synthesize(config, layouter)
    }
}

impl CircuitExt<Fr> for AggregationCircuit {
    fn num_instance(&self) -> Vec<usize> {
        self.builder.num_instance()
    }

    fn instances(&self) -> Vec<Vec<Fr>> {
        self.builder.instances()
    }

    fn accumulator_indices() -> Option<Vec<(usize, usize)>> {
        Some((0..4 * LIMBS).map(|idx| (0, idx)).collect())
    }

    fn selectors(config: &Self::Config) -> Vec<Selector> {
        BaseCircuitBuilder::selectors(config)
    }
}

pub fn load_verify_circuit_degree() -> u32 {
    let path = std::env::var("VERIFY_CONFIG")
        .unwrap_or_else(|_| "./configs/verify_circuit.config".to_string());
    let params: AggregationConfigParams = serde_json::from_reader(
        File::open(path.as_str()).unwrap_or_else(|_| panic!("{path} does not exist")),
    )
    .unwrap();
    params.degree
}
