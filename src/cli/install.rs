use anyhow::{Context, Result};
use console::style;
use dialoguer::Select;
use indicatif::HumanBytes;
use std::collections::HashSet;

use crate::auth::AuthStore;
use crate::compat;
use crate::core::config::Config;
use crate::core::db::Database;
use crate::core::download;
use crate::core::gpu;
use crate::core::huggingface;
use crate::core::manifest::Manifest;
use crate::core::registry::RegistryIndex;
use crate::core::resolver;
use crate::core::store::Store;
use crate::core::symlink;

pub async fn run(id: &str, variant: Option<&str>, dry_run: bool, force: bool) -> Result<()> {
    // Handle hf:owner/repo prefix — direct pull from HuggingFace
    if let Some(repo_id) = id.strip_prefix("hf:") {
        return run_hf_pull(repo_id, variant, dry_run, force).await;
    }

    let config = Config::load()?;
    let index = RegistryIndex::load_or_fetch().await?;
    let db = Database::open()?;

    // Get set of installed model IDs
    let installed_list = db.list_installed(None)?;
    let installed_ids: HashSet<String> = installed_list.iter().map(|m| m.id.clone()).collect();

    // Resolve dependency tree
    let mut plan = resolver::resolve(id, variant, &index, &installed_ids)?;

    // Auto-select variants based on GPU if not specified
    let vram = config
        .gpu
        .as_ref()
        .map(|g| g.vram_mb)
        .or_else(|| gpu::detect().map(|g| g.vram_mb));

    // Interactive variant selection for the primary model (if it has multiple variants
    // and the user didn't specify --variant)
    if variant.is_none() {
        for item in &mut plan.items {
            if item.manifest.id == id && !item.already_installed && item.manifest.variants.len() > 1
            {
                let selected = prompt_variant_selection(&item.manifest, vram)?;
                item.variant_id = Some(selected);
            }
        }
    }

    // Display the plan
    println!(
        "{} Install plan for {}:",
        style("→").cyan(),
        style(id).bold()
    );
    println!();

    let mut total_download: u64 = 0;
    for item in &plan.items {
        let effective_variant = if item.manifest.id == id {
            item.variant_id.as_deref().or(variant)
        } else {
            item.variant_id.as_deref()
        };
        let (_file_name, size, variant_label) =
            get_file_info(&item.manifest, effective_variant, vram);
        let status = if item.already_installed {
            style("installed").green().to_string()
        } else {
            total_download += size;
            style(HumanBytes(size).to_string()).yellow().to_string()
        };
        println!(
            "  {} {} {} {}",
            if item.already_installed {
                style("✓").green()
            } else {
                style("↓").cyan()
            },
            style(&item.manifest.name).bold(),
            style(format!("({})", item.manifest.asset_type)).dim(),
            if let Some(ref v) = variant_label {
                format!("[{}] {}", v, status)
            } else {
                status
            }
        );
    }

    println!();
    if total_download > 0 {
        println!(
            "  Total download: {}",
            style(HumanBytes(total_download)).bold()
        );
    }

    if dry_run {
        println!();
        println!("{}", style("Dry run — nothing downloaded.").dim());
        return Ok(());
    }

    let items_to_install: Vec<_> = plan.items.iter().filter(|i| !i.already_installed).collect();

    if items_to_install.is_empty() {
        println!("{} Everything is already installed.", style("✓").green());
        return Ok(());
    }

    println!();

    // Load auth tokens
    let auth_store = AuthStore::load().unwrap_or_default();
    let store = Store::new(config.store_root());

    // Download each item
    for item in &items_to_install {
        // Only apply the user-specified --variant to the primary model, not to dependencies.
        // Dependencies use their own variant_id set by the resolver, or auto-select by VRAM.
        let effective_variant = if item.manifest.id == id {
            item.variant_id.as_deref().or(variant)
        } else {
            item.variant_id.as_deref()
        };
        let (file_name, size, selected_variant) =
            get_file_info(&item.manifest, effective_variant, vram);
        let url = get_download_url(&item.manifest, effective_variant, vram);
        let sha256 = get_sha256(&item.manifest, effective_variant, vram);

        let store_path = store.path_for(&item.manifest.asset_type, &sha256, &file_name);
        store
            .ensure_dir(&store_path)
            .context("Failed to create store directory")?;

        // Get auth token if needed
        // First check explicit auth in manifest, then fall back to HF token for
        // any huggingface.co URL (some manifests like flux-vae don't declare auth
        // but the CDN still requires a token).
        let auth_token = item
            .manifest
            .auth
            .as_ref()
            .and_then(|a| auth_store.token_for(&a.provider))
            .or_else(|| {
                if url.contains("huggingface.co") {
                    auth_store.token_for("huggingface")
                } else {
                    None
                }
            });

        if item.manifest.auth.as_ref().is_some_and(|a| a.gated) && auth_token.is_none() {
            let provider = &item.manifest.auth.as_ref().unwrap().provider;
            println!(
                "  {} {} requires authentication with {}",
                style("!").yellow(),
                item.manifest.name,
                provider
            );
            println!(
                "    Run: {}",
                style(format!("modl auth {}", provider)).cyan()
            );
            if let Some(ref terms) = item.manifest.auth.as_ref().unwrap().terms_url {
                println!("    Accept terms at: {}", style(terms).underlined());
            }
            anyhow::bail!(
                "Authentication required. Run `modl auth {}` first.",
                provider
            );
        }

        // Download
        if !store_path.exists() || force {
            // Before downloading, check if the file already exists at any target path
            let mut adopted = false;
            if !force {
                for target in &config.targets {
                    if !target.symlink {
                        continue;
                    }
                    let target_path = compat::symlink_path(
                        &target.path,
                        &target.tool_type,
                        &item.manifest.asset_type,
                        &file_name,
                    );
                    if target_path.exists() && !target_path.is_symlink() {
                        // Real file exists at target — check its hash
                        print!(
                            "  {} Found existing {} — verifying... ",
                            style("?").yellow(),
                            target_path.display()
                        );
                        if Store::verify_hash(&target_path, &sha256)? {
                            println!("{}", style("match!").green());
                            // Move file to store, create symlink
                            std::fs::rename(&target_path, &store_path).with_context(|| {
                                format!("Failed to move {} to store", target_path.display())
                            })?;
                            symlink::create(&target_path, &store_path)?;
                            println!("  {} Adopted → {}", style("→").dim(), store_path.display());
                            adopted = true;
                            break;
                        } else {
                            println!("{}", style("hash mismatch, downloading fresh").yellow());
                        }
                    }
                }
            }

            if !adopted {
                match download::download_file(&url, &store_path, Some(size), auth_token.as_deref())
                    .await
                {
                    Ok(()) => {}
                    Err(e) => {
                        // Check for 401 Unauthorized and suggest auth command
                        let err_msg = format!("{}", e);
                        if err_msg.contains("401") || err_msg.contains("Unauthorized") {
                            println!();
                            println!(
                                "  {} This model requires authentication.",
                                style("\u{2717}").red()
                            );
                            let provider = item
                                .manifest
                                .auth
                                .as_ref()
                                .map(|a| a.provider.as_str())
                                .unwrap_or("huggingface");
                            println!(
                                "    Run: {}",
                                style(format!("modl auth {}", provider)).cyan()
                            );
                            if let Some(ref auth) = item.manifest.auth
                                && let Some(ref terms) = auth.terms_url
                            {
                                println!("    Accept terms at: {}", style(terms).underlined());
                            }
                            println!();
                        }
                        return Err(e)
                            .with_context(|| format!("Failed to download {}", item.manifest.name));
                    }
                }

                // Verify hash
                if !Store::verify_hash(&store_path, &sha256)? {
                    let actual_hash =
                        Store::hash_file(&store_path).unwrap_or_else(|_| "unknown".to_string());
                    let actual_size = std::fs::metadata(&store_path).map(|m| m.len()).unwrap_or(0);
                    eprintln!(
                        "  {} SHA256 mismatch for {}",
                        style("✗").red(),
                        item.manifest.name
                    );
                    eprintln!("    expected: {}", sha256);
                    eprintln!("    actual:   {}", actual_hash);
                    eprintln!(
                        "    size:     {} (expected {})",
                        HumanBytes(actual_size),
                        HumanBytes(size)
                    );
                    std::fs::remove_file(&store_path).ok();
                    anyhow::bail!(
                        "SHA256 mismatch for {}. File deleted. Try again.",
                        item.manifest.name
                    );
                }
            }
        } else {
            println!(
                "  {} {} already in store, skipping download",
                style("✓").green(),
                style(&file_name).dim()
            );
        }

        // Create symlinks to all configured targets
        for target in &config.targets {
            if target.symlink {
                let link_path = compat::symlink_path(
                    &target.path,
                    &target.tool_type,
                    &item.manifest.asset_type,
                    &file_name,
                );
                match symlink::create(&link_path, &store_path) {
                    Ok(()) => {
                        println!("  {} Linked → {}", style("→").dim(), link_path.display());
                    }
                    Err(e) => {
                        eprintln!("  {} Symlink failed: {}", style("!").yellow(), e);
                    }
                }
            }
        }

        // Record in database (use actual file size from disk, not manifest estimate)
        let actual_size = std::fs::metadata(&store_path)
            .map(|m| m.len())
            .unwrap_or(size);
        db.insert_installed(&crate::core::db::InstalledModelRecord {
            id: &item.manifest.id,
            name: &item.manifest.name,
            asset_type: &item.manifest.asset_type.to_string(),
            variant: selected_variant.as_deref(),
            sha256: &sha256,
            size: actual_size,
            file_name: &file_name,
            store_path: &store_path.to_string_lossy(),
        })?;

        println!(
            "  {} {}",
            style("✓").green(),
            style(&item.manifest.name).bold()
        );
    }

    println!();
    println!(
        "{} Installed {} successfully.",
        style("✓").green().bold(),
        style(id).bold()
    );

    Ok(())
}

// ── HuggingFace direct pull ─────────────────────────────────────────────

async fn run_hf_pull(
    repo_id: &str,
    variant: Option<&str>,
    dry_run: bool,
    force: bool,
) -> Result<()> {
    let config = Config::load()?;
    let db = Database::open()?;
    let auth_store = AuthStore::load().unwrap_or_default();
    let hf_token = auth_store.token_for("huggingface");

    println!(
        "{} Resolving {} on HuggingFace...",
        style("→").cyan(),
        style(format!("hf:{}", repo_id)).bold()
    );

    // Fetch model info
    let model = huggingface::get_model(repo_id, hf_token.as_deref()).await?;

    // Resolve the best file to download
    let resolved = huggingface::resolve_download(repo_id, variant, hf_token.as_deref()).await?;

    // Guess asset type from HF metadata
    let asset_type_str = huggingface::guess_asset_type(&model, &resolved.filename);
    let asset_type: crate::core::manifest::AssetType = asset_type_str
        .parse()
        .unwrap_or(crate::core::manifest::AssetType::Checkpoint);

    // Build display name from repo ID
    let display_name = repo_id
        .split('/')
        .next_back()
        .unwrap_or(repo_id)
        .to_string();

    // Build a local ID for tracking
    let local_id = format!("hf:{}", repo_id);

    println!();
    println!(
        "  {} {} {}",
        style("↓").cyan(),
        style(&display_name).bold(),
        style(format!("({}) [{}]", asset_type, HumanBytes(resolved.size))).dim(),
    );
    if model.siblings.len() > 1 {
        let model_file_count = model
            .siblings
            .iter()
            .filter(|s| {
                let name = &s.filename;
                [".safetensors", ".ckpt", ".bin", ".pth", ".gguf"]
                    .iter()
                    .any(|ext| name.ends_with(ext))
                    && !name.contains('/')
            })
            .count();
        if model_file_count > 1 {
            println!(
                "  {} Repo has {} model files — downloading: {}",
                style("i").dim(),
                model_file_count,
                style(&resolved.filename).cyan(),
            );
        }
    }
    println!(
        "  {} {}",
        style("URL").dim(),
        style(&resolved.url).dim().underlined(),
    );

    if dry_run {
        println!();
        println!("{}", style("Dry run — nothing downloaded.").dim());
        return Ok(());
    }

    // Check if already installed
    let installed_list = db.list_installed(None)?;
    if !force && installed_list.iter().any(|m| m.id == local_id) {
        println!();
        println!(
            "{} {} is already installed. Use --force to re-download.",
            style("✓").green(),
            style(&display_name).bold()
        );
        return Ok(());
    }

    let store = Store::new(config.store_root());

    // We don't have a pre-known SHA256, so we use a placeholder prefix derived from
    // the repo ID. After download we compute the real hash.
    let temp_prefix = format!("{:0>16x}", {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        repo_id.hash(&mut hasher);
        hasher.finish()
    });

    let store_path = store.path_for(&asset_type, &temp_prefix, &resolved.filename);
    store.ensure_dir(&store_path)?;

    if !store_path.exists() || force {
        download::download_file(
            &resolved.url,
            &store_path,
            if resolved.size > 0 {
                Some(resolved.size)
            } else {
                None
            },
            hf_token.as_deref(),
        )
        .await
        .with_context(|| format!("Failed to download {}", display_name))?;
    } else {
        println!("  {} Already in store", style("✓").green(),);
    }

    // Compute real SHA256
    let sha256 =
        Store::hash_file(&store_path).context("Failed to compute SHA256 of downloaded file")?;

    // Move to content-addressed path if the temp prefix doesn't match
    let real_prefix = &sha256[..16];
    if temp_prefix != real_prefix {
        let real_path = store.path_for(&asset_type, &sha256, &resolved.filename);
        if real_path != store_path {
            store.ensure_dir(&real_path)?;
            if real_path.exists() {
                // Already exists at correct path — remove the temp download
                std::fs::remove_file(&store_path).ok();
            } else {
                std::fs::rename(&store_path, &real_path)
                    .or_else(|_| {
                        // Cross-device fallback
                        std::fs::copy(&store_path, &real_path)?;
                        std::fs::remove_file(&store_path)?;
                        Ok::<(), std::io::Error>(())
                    })
                    .context("Failed to move file to content-addressed path")?;
            }
            // Clean up empty temp dir
            if let Some(parent) = store_path.parent() {
                std::fs::remove_dir(parent).ok();
            }
        }
    }

    let final_path = store.path_for(&asset_type, &sha256, &resolved.filename);
    let actual_size = std::fs::metadata(&final_path)
        .map(|m| m.len())
        .unwrap_or(resolved.size);

    // Create symlinks to configured targets
    for target in &config.targets {
        if target.symlink {
            let link_path = compat::symlink_path(
                &target.path,
                &target.tool_type,
                &asset_type,
                &resolved.filename,
            );
            match symlink::create(&link_path, &final_path) {
                Ok(()) => {
                    println!("  {} Linked → {}", style("→").dim(), link_path.display());
                }
                Err(e) => {
                    eprintln!("  {} Symlink failed: {}", style("!").yellow(), e);
                }
            }
        }
    }

    // Record in database
    db.insert_installed(&crate::core::db::InstalledModelRecord {
        id: &local_id,
        name: &display_name,
        asset_type: &asset_type.to_string(),
        variant: None,
        sha256: &sha256,
        size: actual_size,
        file_name: &resolved.filename,
        store_path: &final_path.to_string_lossy(),
    })?;

    println!();
    println!(
        "{} Installed {} (hf:{}) successfully.",
        style("✓").green().bold(),
        style(&display_name).bold(),
        repo_id,
    );

    Ok(())
}

/// Get file name, size, and variant label for a manifest
fn get_file_info(
    manifest: &Manifest,
    requested_variant: Option<&str>,
    vram: Option<u64>,
) -> (String, u64, Option<String>) {
    if let Some(variant) = select_variant(manifest, requested_variant, vram) {
        (variant.file.clone(), variant.size, Some(variant.id.clone()))
    } else if let Some(ref file) = manifest.file {
        (
            manifest.id.clone() + "." + file.format.as_deref().unwrap_or("safetensors"),
            file.size,
            None,
        )
    } else {
        (format!("{}.safetensors", manifest.id), 0, None)
    }
}

fn get_download_url(
    manifest: &Manifest,
    requested_variant: Option<&str>,
    vram: Option<u64>,
) -> String {
    if let Some(variant) = select_variant(manifest, requested_variant, vram) {
        variant.url.clone()
    } else if let Some(ref file) = manifest.file {
        file.url.clone()
    } else {
        String::new()
    }
}

fn get_sha256(manifest: &Manifest, requested_variant: Option<&str>, vram: Option<u64>) -> String {
    if let Some(variant) = select_variant(manifest, requested_variant, vram) {
        variant.sha256.clone()
    } else if let Some(ref file) = manifest.file {
        file.sha256.clone()
    } else {
        String::new()
    }
}

fn select_variant<'a>(
    manifest: &'a Manifest,
    requested: Option<&str>,
    vram: Option<u64>,
) -> Option<&'a crate::core::manifest::Variant> {
    if manifest.variants.is_empty() {
        return None;
    }

    // If user requested a specific variant
    if let Some(req) = requested {
        return manifest.variants.iter().find(|v| v.id == req);
    }

    // Auto-select based on VRAM
    if let Some(vram_mb) = vram {
        let variant_info: Vec<(String, u64)> = manifest
            .variants
            .iter()
            .map(|v| (v.id.clone(), v.vram_required.unwrap_or(0)))
            .collect();
        if let Some(selected_id) = gpu::select_variant(vram_mb, &variant_info) {
            return manifest.variants.iter().find(|v| v.id == selected_id);
        }
    }

    // Fallback: pick the first (usually smallest or most common)
    manifest.variants.first()
}

/// Show an interactive menu for the user to pick a variant
fn prompt_variant_selection(manifest: &Manifest, vram: Option<u64>) -> Result<String> {
    // Determine which variant would be auto-selected based on VRAM
    let auto_selected = vram.and_then(|vram_mb| {
        let variant_info: Vec<(String, u64)> = manifest
            .variants
            .iter()
            .map(|v| (v.id.clone(), v.vram_required.unwrap_or(0)))
            .collect();
        gpu::select_variant(vram_mb, &variant_info)
    });

    let items: Vec<String> = manifest
        .variants
        .iter()
        .map(|v| {
            let recommended = auto_selected.as_ref().map(|s| s == &v.id).unwrap_or(false);
            let precision = v
                .precision
                .as_ref()
                .map(|p| format!(", {}", p))
                .unwrap_or_default();
            let note = v
                .note
                .as_ref()
                .map(|n| format!(" - {}", n))
                .unwrap_or_default();
            format!(
                "{}  ({}{}){}{}",
                v.id,
                HumanBytes(v.size),
                precision,
                note,
                if recommended {
                    format!("  {}", style("<- recommended for your GPU").dim())
                } else {
                    String::new()
                },
            )
        })
        .collect();

    let default_idx = auto_selected
        .as_ref()
        .and_then(|s| manifest.variants.iter().position(|v| &v.id == s))
        .unwrap_or(0);

    println!(
        "\n  {} {} has multiple variants:",
        style("?").yellow(),
        style(&manifest.name).bold()
    );

    let selection = Select::new()
        .items(&items)
        .default(default_idx)
        .interact()?;

    Ok(manifest.variants[selection].id.clone())
}
