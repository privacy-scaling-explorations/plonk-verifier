//! Transcript traits.

use crate::{
    loader::{native::NativeLoader, Loader},
    {util::arithmetic::CurveAffine, Error},
};

/// Common methods for prover and verifier.
pub trait Transcript<C, L>
where
    C: CurveAffine,
    L: Loader<C>,
{
    /// Returns [`Loader`].
    fn loader(&self) -> &L;

    /// Squeeze a challenge.
    fn squeeze_challenge(&mut self) -> L::LoadedScalar;

    /// Squeeze `n` challenges.
    fn squeeze_n_challenges(&mut self, n: usize) -> Vec<L::LoadedScalar> {
        (0..n).map(|_| self.squeeze_challenge()).collect()
    }

    /// Update with an elliptic curve point.
    fn common_ec_point(&mut self, ec_point: &L::LoadedEcPoint) -> Result<(), Error>;

    /// Update with a scalar.
    fn common_scalar(&mut self, scalar: &L::LoadedScalar) -> Result<(), Error>;
}

/// Transcript for verifier.
pub trait TranscriptRead<C, L>: Transcript<C, L>
where
    C: CurveAffine,
    L: Loader<C>,
{
    /// Read a scalar.
    fn read_scalar(&mut self) -> Result<L::LoadedScalar, Error>;

    /// Read `n` scalar.
    fn read_n_scalars(&mut self, n: usize) -> Result<Vec<L::LoadedScalar>, Error> {
        (0..n).map(|_| self.read_scalar()).collect()
    }

    /// Read an elliptic curve point.
    fn read_ec_point(&mut self) -> Result<L::LoadedEcPoint, Error>;

    /// Read `n` elliptic curve point.
    fn read_n_ec_points(&mut self, n: usize) -> Result<Vec<L::LoadedEcPoint>, Error> {
        (0..n).map(|_| self.read_ec_point()).collect()
    }
}

/// Transcript for prover.
pub trait TranscriptWrite<C: CurveAffine>: Transcript<C, NativeLoader> {
    /// Write a scalar.
    fn write_scalar(&mut self, scalar: C::Scalar) -> Result<(), Error>;

    /// Write an elliptic curve point.
    fn write_ec_point(&mut self, ec_point: C) -> Result<(), Error>;
}
