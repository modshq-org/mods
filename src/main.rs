mod auth;
mod cli;
mod compat;
mod core;
mod ui;

use anyhow::Result;
use clap::Parser;
use cli::Cli;

fn main() {
    // Windows default stack is 1 MB which is too small for our large CLI enum
    // and async state machines. Spawn the real entry point with 8 MB stack.
    let result = std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .name("modl-main".into())
        .spawn(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("failed to build tokio runtime")
                .block_on(async_main())
        })
        .expect("failed to spawn main thread")
        .join()
        .expect("main thread panicked");

    if let Err(e) = result {
        eprintln!("Error: {e:?}");
        std::process::exit(1);
    }
}

async fn async_main() -> Result<()> {
    let cli = Cli::parse();

    // Spawn a non-blocking background update check (at most once per 24h).
    // This never blocks the main command — we just read the result at the end.
    let update_handle = core::update_check::spawn_check();

    let result = cli::run(cli).await;

    // Wait briefly for the background check to finish (it's usually instant
    // since most calls hit a fresh cache). Then print a hint if applicable.
    let _ = tokio::time::timeout(std::time::Duration::from_millis(500), update_handle).await;
    core::update_check::print_if_update_available();

    result
}
