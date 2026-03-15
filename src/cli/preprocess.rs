use anyhow::{Context, Result};
use clap::Subcommand;
use console::style;
use std::path::PathBuf;

use crate::core::job::PreprocessJobSpec;

#[derive(Subcommand)]
pub enum PreprocessMethod {
    /// Extract edge map using Canny (no model needed, pure OpenCV)
    Canny {
        /// Image file(s) or directory
        #[arg(required = true)]
        paths: Vec<String>,
        /// Low threshold for Canny edge detection
        #[arg(long, default_value = "100")]
        low: u32,
        /// High threshold for Canny edge detection
        #[arg(long, default_value = "200")]
        high: u32,
        /// Output path or directory
        #[arg(long, short = 'o')]
        output: Option<String>,
        /// Output result as JSON
        #[arg(long)]
        json: bool,
    },

    /// Extract depth map using Depth Anything V2
    Depth {
        /// Image file(s) or directory
        #[arg(required = true)]
        paths: Vec<String>,
        /// Depth model variant: small (98MB, fast), base (390MB, better)
        #[arg(long, default_value = "small")]
        model: String,
        /// Output path or directory
        #[arg(long, short = 'o')]
        output: Option<String>,
        /// Output result as JSON
        #[arg(long)]
        json: bool,
    },

    /// Extract pose skeleton using DWPose
    Pose {
        /// Image file(s) or directory
        #[arg(required = true)]
        paths: Vec<String>,
        /// Include hand keypoints
        #[arg(long, default_value = "true")]
        include_hands: bool,
        /// Include face landmarks
        #[arg(long, default_value = "true")]
        include_face: bool,
        /// Output path or directory
        #[arg(long, short = 'o')]
        output: Option<String>,
        /// Output result as JSON
        #[arg(long)]
        json: bool,
    },

    /// Extract soft edge map using HED
    Softedge {
        /// Image file(s) or directory
        #[arg(required = true)]
        paths: Vec<String>,
        /// Output path or directory
        #[arg(long, short = 'o')]
        output: Option<String>,
        /// Output result as JSON
        #[arg(long)]
        json: bool,
    },

    /// Extract binary scribble lines from HED
    Scribble {
        /// Image file(s) or directory
        #[arg(required = true)]
        paths: Vec<String>,
        /// Binary threshold (0-255)
        #[arg(long, default_value = "128")]
        threshold: u32,
        /// Output path or directory
        #[arg(long, short = 'o')]
        output: Option<String>,
        /// Output result as JSON
        #[arg(long)]
        json: bool,
    },

    /// Extract clean line art
    Lineart {
        /// Image file(s) or directory
        #[arg(required = true)]
        paths: Vec<String>,
        /// Use coarse (rough) line extraction
        #[arg(long)]
        coarse: bool,
        /// Output path or directory
        #[arg(long, short = 'o')]
        output: Option<String>,
        /// Output result as JSON
        #[arg(long)]
        json: bool,
    },

    /// Extract normal map (derived from depth)
    Normal {
        /// Image file(s) or directory
        #[arg(required = true)]
        paths: Vec<String>,
        /// Depth model variant: small, base
        #[arg(long, default_value = "small")]
        model: String,
        /// Output path or directory
        #[arg(long, short = 'o')]
        output: Option<String>,
        /// Output result as JSON
        #[arg(long)]
        json: bool,
    },
}

pub async fn run(method: PreprocessMethod) -> Result<()> {
    let (method_name, paths, output, json, spec) = match method {
        PreprocessMethod::Canny {
            paths,
            low,
            high,
            output,
            json,
        } => (
            "canny",
            paths.clone(),
            output,
            json,
            PreprocessJobSpec {
                image_paths: paths,
                method: "canny".into(),
                output_dir: None,
                canny_low: low,
                canny_high: high,
                ..Default::default()
            },
        ),
        PreprocessMethod::Depth {
            paths,
            model,
            output,
            json,
        } => (
            "depth",
            paths.clone(),
            output,
            json,
            PreprocessJobSpec {
                image_paths: paths,
                method: "depth".into(),
                output_dir: None,
                depth_model: model,
                ..Default::default()
            },
        ),
        PreprocessMethod::Pose {
            paths,
            include_hands,
            include_face,
            output,
            json,
        } => (
            "pose",
            paths.clone(),
            output,
            json,
            PreprocessJobSpec {
                image_paths: paths,
                method: "pose".into(),
                output_dir: None,
                include_hands,
                include_face,
                ..Default::default()
            },
        ),
        PreprocessMethod::Softedge {
            paths,
            output,
            json,
        } => (
            "softedge",
            paths.clone(),
            output,
            json,
            PreprocessJobSpec {
                image_paths: paths,
                method: "softedge".into(),
                output_dir: None,
                ..Default::default()
            },
        ),
        PreprocessMethod::Scribble {
            paths,
            threshold,
            output,
            json,
        } => (
            "scribble",
            paths.clone(),
            output,
            json,
            PreprocessJobSpec {
                image_paths: paths,
                method: "scribble".into(),
                output_dir: None,
                scribble_threshold: threshold,
                ..Default::default()
            },
        ),
        PreprocessMethod::Lineart {
            paths,
            coarse,
            output,
            json,
        } => {
            let method_str = if coarse { "lineart_coarse" } else { "lineart" };
            (
                "lineart",
                paths.clone(),
                output,
                json,
                PreprocessJobSpec {
                    image_paths: paths,
                    method: method_str.into(),
                    output_dir: None,
                    ..Default::default()
                },
            )
        }
        PreprocessMethod::Normal {
            paths,
            model,
            output,
            json,
        } => (
            "normal",
            paths.clone(),
            output,
            json,
            PreprocessJobSpec {
                image_paths: paths,
                method: "normal".into(),
                output_dir: None,
                depth_model: model,
                ..Default::default()
            },
        ),
    };

    if paths.is_empty() {
        anyhow::bail!(
            "No image paths provided. Usage: modl preprocess {method_name} <image_or_dir> [...]"
        );
    }

    // Verify paths exist
    for p in &paths {
        let path = PathBuf::from(p);
        if !path.exists() {
            anyhow::bail!("Path not found: {p}");
        }
    }

    // Set output_dir if --output was provided
    let mut spec = spec;
    spec.output_dir = output;

    let yaml = serde_yaml::to_string(&spec).context("Failed to serialize preprocess spec")?;

    if !json {
        println!(
            "{} Preprocessing {} image(s) (method: {})...",
            style("→").cyan(),
            paths.len(),
            method_name,
        );
    }

    let result = super::analysis::spawn_analysis_worker("preprocess", &yaml, json).await?;

    if json {
        if let Some(data) = result.result_data {
            println!("{}", serde_json::to_string(&data)?);
        }
    } else if let Some(data) = result.result_data {
        let processed = data.get("processed").and_then(|v| v.as_u64()).unwrap_or(0);
        let errors = data.get("errors").and_then(|v| v.as_u64()).unwrap_or(0);

        if let Some(outputs) = data.get("outputs").and_then(|v| v.as_array()) {
            for entry in outputs {
                if let (Some(input), Some(output)) = (
                    entry.get("input").and_then(|v| v.as_str()),
                    entry.get("output").and_then(|v| v.as_str()),
                ) {
                    let input_name = PathBuf::from(input)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    let output_name = PathBuf::from(output)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    println!("  {} → {}", input_name, style(&output_name).bold());
                }
            }
        }

        if processed > 0 {
            println!(
                "\n  {} Processed {processed} image(s)",
                style("✓").green().bold()
            );
        }
        if errors > 0 {
            println!("  {} {errors} image(s) failed", style("⚠").yellow());
        }
    }

    if !result.success {
        anyhow::bail!("Preprocessing failed");
    }

    Ok(())
}
