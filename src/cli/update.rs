use anyhow::{Context, Result};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};

use crate::core::registry::RegistryIndex;

pub async fn run() -> Result<()> {
    println!("{} Fetching latest registry index...", style("→").cyan());

    let url = RegistryIndex::remote_url();
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner} {msg}")
            .unwrap(),
    );
    pb.set_message("Downloading index.json");
    pb.enable_steady_tick(std::time::Duration::from_millis(80));

    let client = reqwest::Client::new();
    let response = client
        .get(url)
        .send()
        .await
        .context("Failed to fetch registry index")?;

    if !response.status().is_success() {
        pb.finish_with_message("Failed");
        anyhow::bail!(
            "Failed to fetch index: HTTP {}. The registry may not be published yet.",
            response.status()
        );
    }

    let body = response
        .text()
        .await
        .context("Failed to read response body")?;

    let index: RegistryIndex =
        serde_json::from_str(&body).context("Failed to parse registry index")?;

    // Save to local cache
    let path = RegistryIndex::local_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, &body).context("Failed to write index to disk")?;

    pb.finish_and_clear();

    println!(
        "{} Registry updated — {} models available",
        style("✓").green(),
        style(index.items.len()).bold()
    );

    Ok(())
}
