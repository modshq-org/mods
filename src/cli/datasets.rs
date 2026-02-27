use anyhow::{Context, Result};
use comfy_table::{ContentArrangement, Table};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

use crate::core::dataset;
use crate::core::job::CaptionJobSpec;

#[derive(clap::Subcommand)]
pub enum DatasetCommands {
    /// Create a managed dataset from a directory of images
    Create {
        /// Name for the dataset
        name: String,
        /// Source directory containing images (jpg/jpeg/png)
        #[arg(long)]
        from: String,
    },
    /// List all managed datasets
    Ls,
    /// Validate a dataset directory
    Validate {
        /// Dataset name or path to validate
        name_or_path: String,
    },
    /// Auto-caption images using a vision-language model
    Caption {
        /// Dataset name or path
        name_or_path: String,
        /// Captioning model to use
        #[arg(long, default_value = "florence-2", value_parser = ["florence-2", "blip"])]
        model: String,
        /// Re-caption images that already have .txt files
        #[arg(long)]
        overwrite: bool,
    },
}

pub async fn run(command: DatasetCommands) -> Result<()> {
    match command {
        DatasetCommands::Create { name, from } => run_create(&name, &from).await,
        DatasetCommands::Ls => run_list().await,
        DatasetCommands::Validate { name_or_path } => run_validate(&name_or_path).await,
        DatasetCommands::Caption {
            name_or_path,
            model,
            overwrite,
        } => run_caption(&name_or_path, &model, overwrite).await,
    }
}

async fn run_create(name: &str, from: &str) -> Result<()> {
    let from_path = PathBuf::from(from);
    println!(
        "{} Creating dataset '{}' from {}",
        style("→").cyan(),
        style(name).bold(),
        from_path.display()
    );

    let info = dataset::create(name, &from_path)?;

    println!("{} Dataset created", style("✓").green().bold());
    print_dataset_summary(&info);

    if info.image_count < 5 {
        println!(
            "\n{} Only {} images found. Consider adding more for better results (5-20 recommended).",
            style("⚠").yellow(),
            info.image_count
        );
    }

    if info.caption_coverage < 1.0 {
        let uncaptioned = info.image_count - info.captioned_count;
        println!(
            "{} {} images without captions. Add .txt files with the same name for better training.",
            style("ℹ").dim(),
            uncaptioned
        );
    }

    Ok(())
}

async fn run_list() -> Result<()> {
    let datasets = dataset::list()?;

    if datasets.is_empty() {
        println!("No datasets found. Create one with:");
        println!("  mods datasets create <name> --from <dir>");
        return Ok(());
    }

    let mut table = Table::new();
    table.set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec!["Name", "Images", "Captions", "Coverage", "Path"]);

    for ds in &datasets {
        table.add_row(vec![
            ds.name.clone(),
            ds.image_count.to_string(),
            ds.captioned_count.to_string(),
            format!("{:.0}%", ds.caption_coverage * 100.0),
            ds.path.display().to_string(),
        ]);
    }

    println!("{table}");
    Ok(())
}

async fn run_validate(name_or_path: &str) -> Result<()> {
    let path = resolve_dataset_path(name_or_path);

    println!(
        "{} Validating dataset at {}",
        style("→").cyan(),
        path.display()
    );

    let info = dataset::validate(&path)?;

    println!("{} Dataset is valid", style("✓").green().bold());
    print_dataset_summary(&info);

    if info.image_count < 5 {
        println!(
            "\n{} Only {} images. Consider 5-20 for good LoRA quality.",
            style("⚠").yellow(),
            info.image_count
        );
    }

    Ok(())
}

fn print_dataset_summary(info: &dataset::DatasetInfo) {
    println!("  Name:     {}", info.name);
    println!("  Path:     {}", info.path.display());
    println!("  Images:   {}", info.image_count);
    println!(
        "  Captions: {} / {} ({:.0}%)",
        info.captioned_count,
        info.image_count,
        info.caption_coverage * 100.0
    );
}

// ---------------------------------------------------------------------------
// Caption
// ---------------------------------------------------------------------------

async fn run_caption(name_or_path: &str, model: &str, overwrite: bool) -> Result<()> {
    use std::process::{Command, Stdio};
    use std::sync::mpsc;
    use std::thread;

    use crate::core::executor::read_worker_stdout;
    use crate::core::job::EventPayload;
    use crate::core::runtime;
    use crate::core::training::resolve_worker_python_root;

    let path = resolve_dataset_path(name_or_path);

    // Validate dataset exists and has images
    let info = dataset::validate(&path)
        .with_context(|| format!("Could not load dataset '{name_or_path}'"))?;

    let uncaptioned = if overwrite {
        info.image_count
    } else {
        info.image_count - info.captioned_count
    };

    if uncaptioned == 0 {
        println!(
            "{} All {} images already have captions. Nothing to do.",
            style("✓").green().bold(),
            info.image_count
        );
        println!(
            "  Use {} to regenerate existing captions.",
            style("--overwrite").bold()
        );
        return Ok(());
    }

    println!(
        "{} Captioning {} / {} images in '{}' using {}",
        style("→").cyan(),
        uncaptioned,
        info.image_count,
        style(&info.name).bold(),
        style(model).bold(),
    );

    // Build caption spec YAML
    let spec = CaptionJobSpec {
        dataset_path: path.to_string_lossy().to_string(),
        model: model.to_string(),
        overwrite,
    };

    let runtime_root = dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".mods")
        .join("runtime");
    let jobs_dir = runtime_root.join("jobs");
    std::fs::create_dir_all(&jobs_dir)
        .with_context(|| format!("Failed to create jobs dir: {}", jobs_dir.display()))?;

    let job_id = format!("cap-{}", chrono::Utc::now().format("%Y%m%d-%H%M%S"));
    let spec_path = jobs_dir.join(format!("{job_id}.yaml"));
    let yaml = serde_yaml::to_string(&spec).context("Failed to serialize caption spec")?;
    std::fs::write(&spec_path, &yaml)
        .with_context(|| format!("Failed to write spec: {}", spec_path.display()))?;

    // Resolve Python
    let setup = runtime::setup_training(false).await?;
    if !setup.ready {
        anyhow::bail!("Python runtime is not ready. Run `mods train-setup` first.");
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

    // Spawn Python worker
    let mut child = Command::new(&setup.python_path)
        .arg("-m")
        .arg("mods_worker.main")
        .arg("caption")
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
                "Failed to start caption worker using {}",
                setup.python_path.display()
            )
        })?;

    let stdout = child
        .stdout
        .take()
        .context("Failed to capture worker stdout")?;

    // Set up event channel
    let (tx, rx) = mpsc::channel();
    let job_id_clone = job_id.clone();
    thread::spawn(move || {
        read_worker_stdout(stdout, &job_id_clone, tx);
    });

    // Progress bar
    let pb = ProgressBar::new(uncaptioned as u64);
    pb.set_style(
        ProgressStyle::with_template("{spinner:.green} [{bar:30.cyan/dim}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("━╸─"),
    );
    pb.set_message("loading model...");

    let mut captions: Vec<(String, String)> = Vec::new();
    let mut had_error = false;

    // Read events
    for event in rx {
        match event.event {
            EventPayload::Progress {
                step, total_steps, ..
            } => {
                if total_steps > 0 {
                    pb.set_length(total_steps as u64);
                }
                pb.set_position(step as u64);
                pb.set_message("captioning...");
            }
            EventPayload::Log { message, .. } => {
                // Parse caption log lines: "[N/M] filename.jpg (Xs): caption text"
                if let Some(start) = message.find("] ") {
                    let rest = &message[start + 2..];
                    if let Some(colon_pos) = rest.find("): ") {
                        let filename_part = &rest[..colon_pos + 1]; // "filename.jpg (Xs)"
                        let caption = &rest[colon_pos + 3..];
                        // Extract just the filename
                        let filename = filename_part.split(" (").next().unwrap_or(filename_part);
                        captions.push((filename.to_string(), caption.to_string()));
                        pb.set_message(filename.to_string());
                    }
                } else if message.contains("Loading") || message.contains("loaded") {
                    pb.set_message(message.clone());
                }
            }
            EventPayload::Artifact { path, .. } => {
                // Caption .txt artifact written
                pb.println(format!("  {} {}", style("✓").green(), style(&path).dim()));
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
                let msg = message.unwrap_or_else(|| "Captioning completed".into());
                println!("{} {}", style("✓").green().bold(), msg);
            }
            _ => {}
        }
    }

    // Wait for child
    let status = child.wait().context("Failed to wait for caption worker")?;

    // Show caption review
    if !captions.is_empty() {
        println!("\n{}", style("Generated captions:").bold().underlined());
        for (filename, caption) in &captions {
            println!("  {} {}", style(format!("{filename}:")).cyan(), caption);
        }
    }

    // Rescan dataset for updated stats
    if let Ok(updated) = dataset::scan(&path) {
        println!();
        print_dataset_summary(&updated);
    }

    if !status.success() || had_error {
        anyhow::bail!("Caption worker exited with errors");
    }

    Ok(())
}

/// Resolve a name or path to a dataset directory.
/// If it looks like a path (contains / or \), use it directly.
/// Otherwise, look under ~/.mods/datasets/<name>.
fn resolve_dataset_path(name_or_path: &str) -> PathBuf {
    let path = PathBuf::from(name_or_path);
    if path.is_absolute() || name_or_path.contains('/') || name_or_path.contains('\\') {
        path
    } else {
        dirs::home_dir()
            .expect("Could not determine home directory")
            .join(".mods")
            .join("datasets")
            .join(name_or_path)
    }
}
