use anyhow::{Context, Result};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;

use crate::core::executor::read_worker_stdout;
use crate::core::job::EventPayload;
use crate::core::runtime;
use crate::core::training::resolve_worker_python_root;

/// Result of an analysis worker run.
pub struct AnalysisResult {
    pub result_data: Option<serde_json::Value>,
    pub success: bool,
}

/// Spawn a Python worker for an analysis operation and stream progress.
///
/// `worker_command` is the subcommand name (e.g. "score", "detect", "compare").
/// `spec_yaml` is the serialized job spec.
/// `quiet` suppresses progress output (for --json mode).
pub async fn spawn_analysis_worker(
    worker_command: &str,
    spec_yaml: &str,
    quiet: bool,
) -> Result<AnalysisResult> {
    let runtime_root = dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".modl")
        .join("runtime");
    let jobs_dir = runtime_root.join("jobs");
    std::fs::create_dir_all(&jobs_dir)
        .with_context(|| format!("Failed to create jobs dir: {}", jobs_dir.display()))?;

    let job_id = format!(
        "{}-{}",
        worker_command,
        chrono::Utc::now().format("%Y%m%d-%H%M%S")
    );
    let spec_path = jobs_dir.join(format!("{job_id}.yaml"));
    std::fs::write(&spec_path, spec_yaml)
        .with_context(|| format!("Failed to write spec: {}", spec_path.display()))?;

    // Resolve Python
    let setup = runtime::setup_training(false).await?;
    if !setup.ready {
        anyhow::bail!("Python runtime is not ready. Run `modl train setup` first.");
    }

    let worker_root = resolve_worker_python_root()?;
    let mut py_path = worker_root.to_string_lossy().to_string();
    if let Ok(Some(aitk_dir)) = runtime::aitoolkit_path() {
        py_path = format!("{}:{}", py_path, aitk_dir.display());
    }
    if let Ok(current) = std::env::var("PYTHONPATH")
        && !current.trim().is_empty()
    {
        py_path = format!("{}:{}", py_path, current);
    }

    let mut child = Command::new(&setup.python_path)
        .arg("-m")
        .arg("modl_worker.main")
        .arg(worker_command)
        .arg("--config")
        .arg(&spec_path)
        .arg("--job-id")
        .arg(&job_id)
        .env("PYTHONPATH", py_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| {
            format!(
                "Failed to start {} worker using {}",
                worker_command,
                setup.python_path.display()
            )
        })?;

    let stdout = child
        .stdout
        .take()
        .context("Failed to capture worker stdout")?;

    let (tx, rx) = mpsc::channel();
    let job_id_clone = job_id.clone();
    thread::spawn(move || {
        read_worker_stdout(stdout, &job_id_clone, tx);
    });

    let pb = if quiet {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(ProgressStyle::with_template("{spinner:.green} {msg}").unwrap());
        pb.set_message("loading model...");
        pb
    };

    let mut result_data: Option<serde_json::Value> = None;
    let mut had_error = false;

    for event in rx {
        match event.event {
            EventPayload::Progress {
                step, total_steps, ..
            } => {
                if total_steps > 0 {
                    pb.set_length(total_steps as u64);
                    pb.set_style(
                        ProgressStyle::with_template(
                            "{spinner:.green} [{bar:30.cyan/dim}] {pos}/{len} {msg}",
                        )
                        .unwrap()
                        .progress_chars("━╸─"),
                    );
                }
                pb.set_position(step as u64);
                pb.set_message(format!("{}...", worker_command));
            }
            EventPayload::Result { data, .. } => {
                result_data = Some(data);
            }
            EventPayload::Log { message, .. } => {
                if message.contains("Loading")
                    || message.contains("loaded")
                    || message.contains("Downloading")
                {
                    pb.set_message(message);
                }
            }
            EventPayload::Warning { message, .. } => {
                pb.println(format!("  {} {}", style("⚠").yellow(), message));
            }
            EventPayload::Error {
                message,
                recoverable,
                ..
            } => {
                pb.println(format!("  {} {}", style("✗").red(), message));
                if !recoverable {
                    had_error = true;
                }
            }
            EventPayload::Completed { message } => {
                pb.finish_and_clear();
                if !quiet {
                    let msg = message.unwrap_or_else(|| format!("{} completed", worker_command));
                    println!("{} {}", style("✓").green().bold(), msg);
                }
            }
            _ => {}
        }
    }

    let status = child.wait().context("Failed to wait for worker")?;

    Ok(AnalysisResult {
        result_data,
        success: status.success() && !had_error,
    })
}
