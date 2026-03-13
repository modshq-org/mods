//! Centralized path helpers for the modl root directory.
//!
//! Every module that needs `~/.modl` should call `modl_root()` from here
//! instead of constructing the path inline.

use std::path::PathBuf;

/// Returns the modl root directory (`~/.modl`).
///
/// Panics if the home directory cannot be determined (e.g. `$HOME` unset).
/// This is acceptable for a CLI tool — if HOME is missing, nothing will work.
pub fn modl_root() -> PathBuf {
    dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".modl")
}
