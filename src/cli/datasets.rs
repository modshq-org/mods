use anyhow::Result;
use comfy_table::{ContentArrangement, Table};
use console::style;
use std::path::PathBuf;

use crate::core::dataset;

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
    List,
    /// Validate a dataset directory
    Validate {
        /// Dataset name or path to validate
        name_or_path: String,
    },
}

pub async fn run(command: DatasetCommands) -> Result<()> {
    match command {
        DatasetCommands::Create { name, from } => run_create(&name, &from).await,
        DatasetCommands::List => run_list().await,
        DatasetCommands::Validate { name_or_path } => run_validate(&name_or_path).await,
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
