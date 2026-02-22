use anyhow::Result;
use comfy_table::{Cell, Color, Table, presets::UTF8_FULL_CONDENSED};
use console::style;
use indicatif::HumanBytes;

use crate::core::config::Config;
use crate::core::db::Database;

pub async fn run() -> Result<()> {
    let config = Config::load()?;
    let db = Database::open()?;
    let models = db.list_installed(None)?;

    if models.is_empty() {
        println!("No models installed. Nothing to report.");
        return Ok(());
    }

    // Group by type
    let mut by_type: std::collections::BTreeMap<String, Vec<_>> = std::collections::BTreeMap::new();
    for m in &models {
        by_type.entry(m.asset_type.clone()).or_default().push(m);
    }

    println!("{}", style("Disk usage by type:").bold().cyan());
    println!();

    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED);
    table.set_header(vec![
        Cell::new("Type").fg(Color::Cyan),
        Cell::new("Count").fg(Color::Cyan),
        Cell::new("Total Size").fg(Color::Cyan),
    ]);

    let mut grand_total: u64 = 0;

    for (asset_type, items) in &by_type {
        let total: u64 = items.iter().map(|m| m.size).sum();
        grand_total += total;
        table.add_row(vec![
            Cell::new(asset_type),
            Cell::new(items.len().to_string()),
            Cell::new(HumanBytes(total).to_string()),
        ]);
    }

    println!("{table}");
    println!();

    // Store directory size (actual disk usage)
    let store_root = config.store_root().join("store");
    let store_size = if store_root.exists() {
        dir_size(&store_root).unwrap_or(0)
    } else {
        0
    };

    println!("  Store directory:  {}", HumanBytes(store_size));
    println!("  DB tracked total: {}", HumanBytes(grand_total));
    println!("  Models installed: {}", style(models.len()).bold());

    Ok(())
}

fn dir_size(path: &std::path::Path) -> Result<u64> {
    let mut size = 0;
    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let meta = entry.metadata()?;
            if meta.is_dir() {
                size += dir_size(&entry.path())?;
            } else {
                size += meta.len();
            }
        }
    }
    Ok(size)
}
