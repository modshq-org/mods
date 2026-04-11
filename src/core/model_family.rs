//! Model family definitions — capabilities, param counts, defaults, and UI metadata.
//!
//! This module re-exports everything from `models.rs`, which parses `models.toml`
//! (the single source of truth for model metadata). All existing call sites that
//! use `crate::core::model_family::*` continue to work unchanged.
//!
//! The Python `arch_config.py` remains the source of truth for pipeline classes
//! and training flags — this module covers the user-facing / validation side.

// Re-export all types and functions from the TOML-driven models module.
pub use crate::core::models::*;

// Backwards-compat aliases for static slice accessors.
// Old code used `model_family::FAMILIES`, `model_family::LIGHTNING_CONFIGS`, etc.
// New code should use the function form: `families()`, `lightning_configs()`, etc.

use std::sync::LazyLock;

/// Backwards-compatible accessor for `FAMILIES`.
/// Returns a reference to the parsed families from models.toml.
pub static FAMILIES: LazyLock<Vec<ModelFamily>> = LazyLock::new(|| families().to_vec());

/// Backwards-compatible accessor for `LIGHTNING_CONFIGS`.
pub static LIGHTNING_CONFIGS: LazyLock<Vec<LightningConfig>> =
    LazyLock::new(|| lightning_configs().to_vec());

/// Backwards-compatible accessor for `CONTROLNET_SUPPORT`.
pub static CONTROLNET_SUPPORT: LazyLock<Vec<ControlNetSupport>> =
    LazyLock::new(|| controlnet_support_list().to_vec());

/// Backwards-compatible accessor for `STYLE_REF_SUPPORT`.
pub static STYLE_REF_SUPPORT: LazyLock<Vec<StyleRefSupport>> =
    LazyLock::new(|| style_ref_support_list().to_vec());
