use anyhow::Result;
use console::style;
use indicatif::HumanBytes;

use crate::core::db::Database;
use crate::core::registry::RegistryIndex;

pub async fn run(id: &str) -> Result<()> {
    let index = RegistryIndex::load()?;
    let db = Database::open()?;

    let manifest = index.find(id).ok_or_else(|| {
        anyhow::anyhow!("'{}' not found in registry. Run `mods update` first?", id)
    })?;

    let installed = db.is_installed(id)?;

    // Header
    println!(
        "{} {}",
        style(&manifest.name).bold().cyan(),
        if installed {
            style("[installed]").green().to_string()
        } else {
            String::new()
        }
    );
    println!(
        "  {} · {}",
        style(&manifest.asset_type).dim(),
        style(id).dim()
    );
    println!();

    // Description
    if let Some(ref desc) = manifest.description {
        println!("{}", desc.trim());
        println!();
    }

    // Metadata
    if let Some(ref author) = manifest.author {
        println!("  Author:       {}", author);
    }
    if let Some(ref arch) = manifest.architecture {
        println!("  Architecture: {}", arch);
    }
    if let Some(ref license) = manifest.license {
        println!("  License:      {}", license);
    }
    if let Some(ref homepage) = manifest.homepage {
        println!("  Homepage:     {}", style(homepage).underlined());
    }
    if let Some(rating) = manifest.rating {
        println!("  Rating:       {:.1} / 5.0", rating);
    }
    if !manifest.tags.is_empty() {
        println!("  Tags:         {}", manifest.tags.join(", "));
    }

    // Variants
    if !manifest.variants.is_empty() {
        println!();
        println!("  {}", style("Variants:").bold());
        for v in &manifest.variants {
            let vram = v
                .vram_required
                .map(|mb| format!(" ({}+ MB VRAM)", mb))
                .unwrap_or_default();
            println!(
                "    {} — {} {}{}",
                style(&v.id).cyan(),
                HumanBytes(v.size),
                v.precision.as_deref().unwrap_or(""),
                style(vram).dim()
            );
            if let Some(ref note) = v.note {
                println!("      {}", style(note).dim());
            }
        }
    }

    // Single file
    if let Some(ref file) = manifest.file {
        println!();
        println!("  Size: {}", HumanBytes(file.size));
    }

    // Dependencies
    if !manifest.requires.is_empty() {
        println!();
        println!("  {}", style("Dependencies:").bold());
        for dep in &manifest.requires {
            let reason = dep
                .reason
                .as_deref()
                .map(|r| format!(" — {}", r))
                .unwrap_or_default();
            println!(
                "    {} ({}){}",
                style(&dep.id).cyan(),
                dep.dep_type,
                style(reason).dim()
            );
        }
    }

    // Auth
    if let Some(ref auth) = manifest.auth
        && auth.gated
    {
        println!();
        println!(
            "  {} Requires {} authentication",
            style("!").yellow(),
            style(&auth.provider).bold()
        );
        if let Some(ref url) = auth.terms_url {
            println!("    Accept terms: {}", style(url).underlined());
        }
    }

    // LoRA-specific
    if !manifest.base_models.is_empty() {
        println!();
        println!("  Compatible with: {}", manifest.base_models.join(", "));
    }
    if !manifest.trigger_words.is_empty() {
        println!("  Trigger words:   {}", manifest.trigger_words.join(", "));
    }
    if let Some(w) = manifest.recommended_weight {
        println!("  Recommended weight: {}", w);
    }

    // Defaults
    if let Some(ref defaults) = manifest.defaults {
        println!();
        println!("  {}", style("Recommended settings:").bold());
        if let Some(steps) = defaults.steps {
            println!("    Steps:     {}", steps);
        }
        if let Some(cfg) = defaults.cfg {
            println!("    CFG:       {}", cfg);
        }
        if let Some(ref sampler) = defaults.sampler {
            println!("    Sampler:   {}", sampler);
        }
        if let Some(ref scheduler) = defaults.scheduler {
            println!("    Scheduler: {}", scheduler);
        }
    }

    // Installed status
    if installed {
        let models = db.list_installed(None)?;
        if let Some(m) = models.iter().find(|m| m.id == id) {
            println!();
            println!("  {}", style("Installed:").bold().green());
            if let Some(ref v) = m.variant {
                println!("    Variant:  {}", v);
            }
            println!("    Size:     {}", HumanBytes(m.size));
            println!("    Path:     {}", m.store_path);
        }
    }

    Ok(())
}
