use anyhow::Result;
use comfy_table::{Cell, Color, Table, presets::UTF8_FULL_CONDENSED};
use console::style;
use indicatif::HumanBytes;

use crate::core::db::Database;

pub async fn run(type_filter: Option<&str>) -> Result<()> {
    let db = Database::open()?;
    let models = db.list_installed(type_filter)?;

    if models.is_empty() {
        if let Some(t) = type_filter {
            println!("No installed models of type '{}'.", t);
        } else {
            println!("No models installed yet.");
            println!(
                "  Run {} to get started.",
                style("mods install flux-dev").cyan()
            );
        }
        return Ok(());
    }

    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED);
    table.set_header(vec![
        Cell::new("Name").fg(Color::Cyan),
        Cell::new("Type").fg(Color::Cyan),
        Cell::new("Variant").fg(Color::Cyan),
        Cell::new("Size").fg(Color::Cyan),
        Cell::new("ID").fg(Color::Cyan),
    ]);

    let mut total_size: u64 = 0;

    for model in &models {
        total_size += model.size;
        table.add_row(vec![
            Cell::new(&model.name),
            Cell::new(&model.asset_type),
            Cell::new(model.variant.as_deref().unwrap_or("—")),
            Cell::new(HumanBytes(model.size).to_string()),
            Cell::new(&model.id).fg(Color::DarkGrey),
        ]);
    }

    println!("{table}");
    println!();
    println!(
        "  {} models, {} total",
        style(models.len()).bold(),
        style(HumanBytes(total_size)).bold()
    );

    Ok(())
}
