#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::upper_case_acronyms)]

pub mod cost;
pub mod loader;
pub mod pcs;
pub mod system;
pub mod util;
pub mod verifier;

#[derive(Clone, Debug)]
pub enum Error {
    InvalidInstances,
    InvalidLinearization,
    InvalidQuery(util::protocol::Query),
    InvalidChallenge(usize),
    AssertionFailure(String),
    Transcript(std::io::ErrorKind, String),
}

#[derive(Clone, Debug)]
pub struct Protocol<C, L = loader::native::NativeLoader>
where
    C: util::arithmetic::CurveAffine,
    L: loader::Loader<C>,
{
    // Common description
    pub domain: util::arithmetic::Domain<C::Scalar>,
    pub preprocessed: Vec<L::LoadedEcPoint>,
    pub num_instance: Vec<usize>,
    pub num_witness: Vec<usize>,
    pub num_challenge: Vec<usize>,
    pub evaluations: Vec<util::protocol::Query>,
    pub queries: Vec<util::protocol::Query>,
    pub quotient: util::protocol::QuotientPolynomial<C::Scalar>,
    // Minor customization
    pub transcript_initial_state: Option<L::LoadedScalar>,
    pub instance_committing_key: Option<util::protocol::InstanceCommittingKey<C>>,
    pub linearization: Option<util::protocol::LinearizationStrategy>,
    pub accumulator_indices: Vec<Vec<(usize, usize)>>,
}
