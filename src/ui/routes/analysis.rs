use axum::{Json, http::StatusCode, response::IntoResponse};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

use crate::core::db::Database;
use crate::core::job::{RemoveBgJobSpec, UpscaleJobSpec};

fn modl_root() -> PathBuf {
    dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".modl")
}

fn output_dir_today() -> String {
    let date = chrono::Local::now().format("%Y-%m-%d");
    let dir = modl_root().join("outputs").join(date.to_string());
    let _ = std::fs::create_dir_all(&dir);
    dir.to_string_lossy().to_string()
}

fn resolve_model_path(model_id: &str, asset_type: &str) -> Option<String> {
    let db = Database::open().ok()?;
    let installed = db.list_installed(None).ok()?;
    installed
        .iter()
        .find(|m| (m.id == model_id || m.name == model_id) && m.asset_type == asset_type)
        .map(|m| m.store_path.clone())
}

/// Resolve an image path from a relative outputs path to absolute.
fn resolve_image_path(rel_path: &str) -> Option<PathBuf> {
    let abs = modl_root().join(rel_path);
    if abs.exists() {
        Some(abs)
    } else {
        // Try as absolute path
        let p = PathBuf::from(rel_path);
        if p.exists() { Some(p) } else { None }
    }
}

// ---------------------------------------------------------------------------
// Upscale
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct UpscaleRequest {
    /// Relative path (outputs/2026-03-10/file.png) or absolute
    pub image_path: String,
    #[serde(default = "default_scale")]
    pub scale: u32,
}

fn default_scale() -> u32 {
    4
}

#[derive(Serialize)]
pub struct AnalysisResponse {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub async fn api_upscale(Json(req): Json<UpscaleRequest>) -> impl IntoResponse {
    let abs_path = match resolve_image_path(&req.image_path) {
        Some(p) => p,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(AnalysisResponse {
                    status: "error".into(),
                    output_path: None,
                    error: Some(format!("Image not found: {}", req.image_path)),
                }),
            )
                .into_response();
        }
    };

    let model_path = match resolve_model_path("realesrgan-x4plus", "upscaler")
        .or_else(|| resolve_model_path("4x-ultrasharp", "upscaler"))
    {
        Some(p) => p,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(AnalysisResponse {
                    status: "error".into(),
                    output_path: None,
                    error: Some(
                        "No upscaler model installed. Run `modl pull realesrgan-x4plus`".into(),
                    ),
                }),
            )
                .into_response();
        }
    };

    let out_dir = output_dir_today();
    let spec = UpscaleJobSpec {
        image_paths: vec![abs_path.to_string_lossy().to_string()],
        output_dir: out_dir.clone(),
        scale: req.scale,
        model_path: Some(model_path),
    };

    let yaml = match serde_yaml::to_string(&spec) {
        Ok(y) => y,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(AnalysisResponse {
                    status: "error".into(),
                    output_path: None,
                    error: Some(format!("Failed to serialize spec: {e}")),
                }),
            )
                .into_response();
        }
    };

    eprintln!("[analysis] upscale {}x: {}", req.scale, abs_path.display());

    match crate::cli::analysis::spawn_analysis_worker("upscale", &yaml, true).await {
        Ok(result) if result.success => {
            // Find the output file — it's in out_dir with the same stem + suffix
            let output_path = find_newest_file(&out_dir);
            let rel = output_path.as_ref().map(|p| {
                p.strip_prefix(&modl_root().to_string_lossy().to_string())
                    .unwrap_or(p)
                    .trim_start_matches('/')
                    .to_string()
            });

            // Register artifact with metadata inherited from source image
            if let Some(out_abs) = &output_path {
                register_derived_artifact(
                    out_abs,
                    &abs_path.to_string_lossy(),
                    &format!("upscale_{}x", req.scale),
                );
            }

            Json(AnalysisResponse {
                status: "completed".into(),
                output_path: rel,
                error: None,
            })
            .into_response()
        }
        Ok(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(AnalysisResponse {
                status: "error".into(),
                output_path: None,
                error: Some("Upscale failed".into()),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(AnalysisResponse {
                status: "error".into(),
                output_path: None,
                error: Some(format!("{e:#}")),
            }),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// Remove background
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct RemoveBgRequest {
    pub image_path: String,
}

pub async fn api_remove_bg(Json(req): Json<RemoveBgRequest>) -> impl IntoResponse {
    let abs_path = match resolve_image_path(&req.image_path) {
        Some(p) => p,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(AnalysisResponse {
                    status: "error".into(),
                    output_path: None,
                    error: Some(format!("Image not found: {}", req.image_path)),
                }),
            )
                .into_response();
        }
    };

    let model_path = match resolve_model_path("birefnet-dis", "segmentation") {
        Some(p) => p,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(AnalysisResponse {
                    status: "error".into(),
                    output_path: None,
                    error: Some("BiRefNet not installed. Run `modl pull birefnet-dis`".into()),
                }),
            )
                .into_response();
        }
    };

    let out_dir = output_dir_today();
    let spec = RemoveBgJobSpec {
        image_paths: vec![abs_path.to_string_lossy().to_string()],
        output_dir: out_dir.clone(),
        model_path: Some(model_path),
    };

    let yaml = match serde_yaml::to_string(&spec) {
        Ok(y) => y,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(AnalysisResponse {
                    status: "error".into(),
                    output_path: None,
                    error: Some(format!("Failed to serialize spec: {e}")),
                }),
            )
                .into_response();
        }
    };

    eprintln!("[analysis] remove-bg: {}", abs_path.display());

    match crate::cli::analysis::spawn_analysis_worker("remove-bg", &yaml, true).await {
        Ok(result) if result.success => {
            let output_path = find_newest_file(&out_dir);
            let rel = output_path.as_ref().map(|p| {
                p.strip_prefix(&modl_root().to_string_lossy().to_string())
                    .unwrap_or(p)
                    .trim_start_matches('/')
                    .to_string()
            });

            // Register artifact with metadata inherited from source image
            if let Some(out_abs) = &output_path {
                register_derived_artifact(out_abs, &abs_path.to_string_lossy(), "remove_bg");
            }

            Json(AnalysisResponse {
                status: "completed".into(),
                output_path: rel,
                error: None,
            })
            .into_response()
        }
        Ok(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(AnalysisResponse {
                status: "error".into(),
                output_path: None,
                error: Some("Background removal failed".into()),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(AnalysisResponse {
                status: "error".into(),
                output_path: None,
                error: Some(format!("{e:#}")),
            }),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Register a derived image (upscale, remove-bg) as a DB artifact,
/// inheriting metadata from the source image's artifact record.
fn register_derived_artifact(output_abs_path: &str, source_abs_path: &str, operation: &str) {
    let db = match Database::open() {
        Ok(db) => db,
        Err(e) => {
            eprintln!("[analysis] Failed to open DB for artifact registration: {e}");
            return;
        }
    };

    // Look up source image's metadata
    let source_meta = db
        .find_artifact_by_path(source_abs_path)
        .ok()
        .flatten()
        .and_then(|a| a.metadata);

    // Build derived metadata: inherit source fields + add provenance
    let mut meta = if let Some(ref raw) = source_meta {
        serde_json::from_str::<serde_json::Value>(raw).unwrap_or_default()
    } else {
        serde_json::json!({})
    };
    if let Some(obj) = meta.as_object_mut() {
        obj.insert(
            "derived_from".to_string(),
            serde_json::json!(source_abs_path),
        );
        obj.insert("operation".to_string(), serde_json::json!(operation));
    }

    // Compute file hash + size
    let (sha256, size_bytes) = match std::fs::read(output_abs_path) {
        Ok(bytes) => {
            let hash = format!("{:x}", Sha256::digest(&bytes));
            (hash, bytes.len() as u64)
        }
        Err(_) => (String::new(), 0),
    };

    let artifact_id = format!("derived:{}:{}", operation, &sha256[..16.min(sha256.len())]);
    let meta_str = meta.to_string();

    if let Err(e) = db.insert_artifact(
        &artifact_id,
        None,
        "image",
        output_abs_path,
        &sha256,
        size_bytes,
        Some(&meta_str),
    ) {
        eprintln!("[analysis] Failed to register derived artifact: {e}");
    }
}

/// Find the most recently modified file in a directory.
fn find_newest_file(dir: &str) -> Option<String> {
    let entries = std::fs::read_dir(dir).ok()?;
    entries
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.ends_with(".png") || name.ends_with(".jpg") || name.ends_with(".webp")
        })
        .max_by_key(|e| {
            e.metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        })
        .map(|e| e.path().to_string_lossy().to_string())
}
