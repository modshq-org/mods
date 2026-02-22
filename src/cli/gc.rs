use anyhow::Result;
use console::style;
use dialoguer::Confirm;
use indicatif::HumanBytes;
use std::collections::HashSet;
use std::path::PathBuf;

use crate::core::config::Config;
use crate::core::db::Database;

pub async fn run() -> Result<()> {
    let config = Config::load()?;
    let db = Database::open()?;
    let models = db.list_installed(None)?;

    let store_root = config.store_root().join("store");
    if !store_root.exists() {
        println!("Store directory doesn't exist. Nothing to clean.");
        return Ok(());
    }

    // Collect all referenced store paths
    let referenced: HashSet<PathBuf> = models
        .iter()
        .map(|m| PathBuf::from(&m.store_path))
        .collect();

    // Walk store and find unreferenced files
    let mut unreferenced: Vec<(PathBuf, u64)> = Vec::new();
    walk_files(&store_root, &referenced, &mut unreferenced)?;

    if unreferenced.is_empty() {
        println!(
            "{} Store is clean — no unreferenced files.",
            style("✓").green()
        );
        return Ok(());
    }

    let total: u64 = unreferenced.iter().map(|(_, s)| s).sum();

    println!(
        "{} Found {} unreferenced file{} ({})",
        style("!").yellow(),
        unreferenced.len(),
        if unreferenced.len() == 1 { "" } else { "s" },
        HumanBytes(total)
    );

    for (path, size) in &unreferenced {
        println!(
            "  {} {} ({})",
            style("×").red(),
            path.display(),
            HumanBytes(*size)
        );
    }

    println!();
    let confirm = Confirm::new()
        .with_prompt(format!(
            "Delete {} files and reclaim {}?",
            unreferenced.len(),
            HumanBytes(total)
        ))
        .default(false)
        .interact()?;

    if !confirm {
        println!("Cancelled.");
        return Ok(());
    }

    let mut freed: u64 = 0;
    for (path, size) in &unreferenced {
        if std::fs::remove_file(path).is_ok() {
            freed += size;
            // Try to remove empty parent dirs
            if let Some(parent) = path.parent() {
                let _ = std::fs::remove_dir(parent); // Silently fails if not empty
            }
        }
    }

    println!(
        "{} Reclaimed {}",
        style("✓").green().bold(),
        style(HumanBytes(freed)).bold()
    );

    Ok(())
}

fn walk_files(
    dir: &std::path::Path,
    referenced: &HashSet<PathBuf>,
    unreferenced: &mut Vec<(PathBuf, u64)>,
) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_files(&path, referenced, unreferenced)?;
        } else if !referenced.contains(&path) {
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            unreferenced.push((path, size));
        }
    }
    Ok(())
}
