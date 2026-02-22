use anyhow::Result;
use console::style;

use crate::core::config::Config;
use crate::core::db::Database;

pub async fn run(id: &str, force: bool) -> Result<()> {
    let db = Database::open()?;
    let config = Config::load()?;

    if !db.is_installed(id)? {
        anyhow::bail!("'{}' is not installed.", id);
    }

    // Check for dependents (items that depend on this one)
    // For now, simple check — a future improvement would query the dependencies table
    if !force {
        println!("{} Checking for dependent models...", style("→").cyan());
        // TODO: Query dependencies table for items that require this id
        // For now, allow uninstall freely
    }

    // Get installed model info before removing
    let models = db.list_installed(None)?;
    let model = models.iter().find(|m| m.id == id);

    if let Some(m) = model {
        // Remove symlinks from all targets
        for target in &config.targets {
            if target.symlink {
                let link_path = crate::compat::symlink_path(
                    &target.path,
                    &target.tool_type,
                    &parse_asset_type(&m.asset_type),
                    &m.file_name,
                );
                if link_path.is_symlink() {
                    std::fs::remove_file(&link_path).ok();
                    println!(
                        "  {} Removed symlink: {}",
                        style("×").red(),
                        link_path.display()
                    );
                }
            }
        }

        println!(
            "  {} Marked {} as uninstalled",
            style("×").red(),
            style(&m.name).bold()
        );
        println!(
            "  {} Store file kept — run {} to reclaim space",
            style("i").dim(),
            style("mods gc").cyan()
        );
    }

    db.remove_installed(id)?;

    println!();
    println!("{} Uninstalled '{}'.", style("✓").green(), id);

    Ok(())
}

fn parse_asset_type(s: &str) -> crate::core::manifest::AssetType {
    match s {
        "checkpoint" => crate::core::manifest::AssetType::Checkpoint,
        "lora" => crate::core::manifest::AssetType::Lora,
        "vae" => crate::core::manifest::AssetType::Vae,
        "text_encoder" => crate::core::manifest::AssetType::TextEncoder,
        "controlnet" => crate::core::manifest::AssetType::Controlnet,
        "upscaler" => crate::core::manifest::AssetType::Upscaler,
        "embedding" => crate::core::manifest::AssetType::Embedding,
        "ipadapter" => crate::core::manifest::AssetType::Ipadapter,
        _ => crate::core::manifest::AssetType::Checkpoint,
    }
}
