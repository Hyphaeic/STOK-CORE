//! # Cumulative Event Function (CEF) and Temporal Event Function (TEF)
//!
//! Utilities for tracking event probabilities over time.
//!
//! Used in high-dimensional STOK factorization (Equation [23]).

use crate::stok::STOKKernel;
use burn::prelude::*;

/// Cumulative Event Function
///
/// Tracks probability that an event occurs by time t.
///
/// κ(z, t_f) = Σ_{τ=0}^{t_f} Σ_{z_f} η(z_f, τ | z)
///
/// κ̄(z, t_f) = 1 - κ(z, t_f) = probability no event in [0, t_f]
pub struct CumulativeEventFunction<B: Backend> {
    /// κ(z, t) for each state and time
    /// Shape: [n_states, max_time]
    pub kappa: Tensor<B, 2>,

    /// Complement: κ̄(z, t) = 1 - κ(z, t)
    /// Shape: [n_states, max_time]
    pub kappa_bar: Tensor<B, 2>,

    /// Metadata
    pub n_states: usize,
    pub max_time: usize,
}

impl<B: Backend> CumulativeEventFunction<B> {
    /// Compute CEF from a STOK
    ///
    /// # Arguments
    ///
    /// * `stok` - STOK kernel to compute CEF from
    ///
    /// # Returns
    ///
    /// Cumulative event function with κ and κ̄
    pub fn from_stok(stok: &STOKKernel<B>) -> Self {
        let s = stok.n_states();
        let t = stok.max_time();
        let device = stok.device();

        // κ(x, t_f) = Σ_{τ=0}^{t_f} Σ_{x_f} η(x_f, τ | x)
        let eta_combined = stok.combined_stok(); // [S, S, T]

        // Sum over final states first: [S, S, T] -> [S, T]
        let eta_summed_states = eta_combined.sum_dim(1).squeeze::<2>();

        // Cumulative sum over time
        let mut kappa = Tensor::zeros([s, t], &device);
        let mut cumsum = Tensor::zeros([s], &device);

        for time in 0..t {
            let eta_t = eta_summed_states
                .clone()
                .slice([0..s, time..(time + 1)])
                .reshape([s]);
            cumsum = cumsum + eta_t;
            kappa = kappa.slice_assign([0..s, time..(time + 1)], cumsum.clone().reshape([s, 1]));
        }

        // Complement
        let kappa_bar = Tensor::ones([s, t], &device) - kappa.clone();

        Self {
            kappa,
            kappa_bar,
            n_states: s,
            max_time: t,
        }
    }

    /// Query κ(z, t)
    ///
    /// # Arguments
    ///
    /// * `state` - State index
    /// * `time` - Time index
    ///
    /// # Returns
    ///
    /// Cumulative event probability
    pub fn event_probability(&self, state: usize, time: usize) -> f32 {
        self.kappa
            .clone()
            .slice([state..(state + 1), time..(time + 1)])
            .into_scalar()
            .elem()
    }

    /// Query κ̄(z, t)
    ///
    /// # Arguments
    ///
    /// * `state` - State index
    /// * `time` - Time index
    ///
    /// # Returns
    ///
    /// Probability no event by time t
    pub fn no_event_probability(&self, state: usize, time: usize) -> f32 {
        self.kappa_bar
            .clone()
            .slice([state..(state + 1), time..(time + 1)])
            .into_scalar()
            .elem()
    }
}

/// Temporal Event Function
///
/// Probability that first event occurs exactly at time t_f.
///
/// ξ(t_f | s) = κ(s, t_f) - κ(s, t_f - 1)
pub struct TemporalEventFunction<B: Backend> {
    /// ξ(t | s) for each state and time
    /// Shape: [n_states, max_time]
    pub xi: Tensor<B, 2>,

    /// Metadata
    pub n_states: usize,
    pub max_time: usize,
}

impl<B: Backend> TemporalEventFunction<B> {
    /// Compute TEF from CEF
    ///
    /// Implements: ξ(t_f | s) = κ(s, t_f) - κ(s, t_f - 1)
    ///
    /// # Arguments
    ///
    /// * `cef` - Cumulative event function
    ///
    /// # Returns
    ///
    /// Temporal event function
    pub fn from_cef(cef: &CumulativeEventFunction<B>) -> Self {
        let s = cef.n_states;
        let t = cef.max_time;
        let device = cef.kappa.device();

        let mut xi = Tensor::zeros([s, t], &device);

        // ξ(t=0) = κ(t=0)
        let xi_0 = cef.kappa.clone().slice([0..s, 0..1]);
        xi = xi.slice_assign([0..s, 0..1], xi_0);

        // ξ(t) = κ(t) - κ(t-1) for t > 0
        for time in 1..t {
            let kappa_t = cef.kappa.clone().slice([0..s, time..(time + 1)]);
            let kappa_t_minus_1 = cef.kappa.clone().slice([0..s, (time - 1)..time]);
            let xi_t = kappa_t - kappa_t_minus_1;
            xi = xi.slice_assign([0..s, time..(time + 1)], xi_t);
        }

        Self {
            xi,
            n_states: s,
            max_time: t,
        }
    }

    /// Query ξ(t | s)
    ///
    /// # Arguments
    ///
    /// * `state` - State index
    /// * `time` - Time index
    ///
    /// # Returns
    ///
    /// Probability first event at time t
    pub fn event_at_time(&self, state: usize, time: usize) -> f32 {
        self.xi
            .clone()
            .slice([state..(state + 1), time..(time + 1)])
            .into_scalar()
            .elem()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{default_device, DefaultBackend};
    use crate::stok::STOKKernel;
    use super::super::spk::StatePredictionKernel;
    use burn::prelude::Tensor;

    #[test]
    fn test_spk_creation() {
        let device = default_device();
        let transition: Tensor<DefaultBackend, 2> = Tensor::eye(3, &device);
        let spk = StatePredictionKernel::new(transition, 5);

        assert_eq!(spk.n_states(), 3);
        assert_eq!(spk.max_time(), 5);
    }

    #[test]
    fn test_cef_from_stok() {
        let device = default_device();
        let stok: STOKKernel<DefaultBackend> = STOKKernel::new(3, 5, &device);

        let cef = CumulativeEventFunction::from_stok(&stok);

        assert_eq!(cef.n_states, 3);
        assert_eq!(cef.max_time, 5);
    }

    #[test]
    fn test_tef_from_cef() {
        let device = default_device();
        let stok: STOKKernel<DefaultBackend> = STOKKernel::new(3, 5, &device);
        let cef = CumulativeEventFunction::from_stok(&stok);
        let tef = TemporalEventFunction::from_cef(&cef);

        assert_eq!(tef.n_states, 3);
        assert_eq!(tef.max_time, 5);
    }
}
