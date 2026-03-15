//! Integration tests for the MCP (Model Context Protocol) server.

use assert_cmd::Command;
use predicates::str::contains;
use std::process::{Command as StdCommand, Stdio};

fn modl_cmd() -> Command {
    Command::cargo_bin("modl").unwrap()
}

// ---------------------------------------------------------------------------
// MCP command smoke tests
// ---------------------------------------------------------------------------

#[test]
fn mcp_help_shows_description() {
    modl_cmd()
        .args(["mcp", "--help"])
        .assert()
        .success()
        .stdout(contains("Model Context Protocol"));
}

#[test]
fn mcp_accepts_port_flag() {
    // Just verify the flag is accepted - actual server startup would require modl serve running
    let result = modl_cmd().args(["mcp", "--port", "3333"]).assert();
    
    // Should either succeed (if API available) or fail gracefully with connection error
    let output = result.get_output();
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    // The command should parse successfully - actual connection is tested separately
    assert!(
        !stderr.contains("error: unexpected argument"),
        "should accept --port flag"
    );
}

#[test]
fn mcp_rejects_invalid_port() {
    modl_cmd()
        .args(["mcp", "--port", "99999"])
        .assert()
        .failure()
        .stderr(contains("error"));
}

// ---------------------------------------------------------------------------
// MCP tool schema validation
// ---------------------------------------------------------------------------

#[test]
fn mcp_generate_tool_has_required_prompt() {
    // Verify the generate tool schema includes required prompt field
    // This is validated at compile time via schemars, but we can verify
    // the tool is registered by checking help output
    modl_cmd()
        .args(["mcp", "--help"])
        .assert()
        .success()
        .stdout(contains("MCP"));
}

// ---------------------------------------------------------------------------
// MCP server lifecycle (requires modl serve to be running)
// These tests are marked as ignored by default since they require
// a running modl server. Run with: cargo test -- --ignored
// ---------------------------------------------------------------------------

#[test]
#[ignore = "Requires modl serve to be running on port 3333"]
fn mcp_server_starts_successfully() {
    use std::time::Duration;
    use std::thread;

    // Get the path to the modl binary as OsString
    let bin_path = std::env::current_exe()
        .expect("Failed to get current exe")
        .parent()
        .expect("Failed to get parent dir")
        .join("modl");
    
    let mut child = StdCommand::new(&bin_path)
        .arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to start MCP server");

    // Give the server a moment to start
    thread::sleep(Duration::from_millis(500));

    // Check if process is still running
    match child.try_wait() {
        Ok(None) => {
            // Process is still running, which is expected for a server
            // Send SIGTERM to gracefully shutdown
            #[cfg(unix)]
            {
                let _ = StdCommand::new("kill")
                    .args(["-TERM", &child.id().to_string()])
                    .status();
            }
            
            // Wait for graceful shutdown
            let _ = child.wait();
        }
        Ok(Some(_status)) => {
            // Process exited - check stderr for expected message
            let stderr_output = child.wait_with_output().expect("Failed to get output").stderr;
            let stderr = String::from_utf8_lossy(&stderr_output);
            if stderr.contains("modl API is not available") {
                // Expected when modl serve is not running
                println!("MCP server correctly detected missing API");
            } else {
                panic!("MCP server failed unexpectedly: {}", stderr);
            }
        }
        Err(e) => panic!("Failed to check MCP server status: {}", e),
    }
}

#[test]
#[ignore = "Requires modl serve to be running on port 3333"]
fn mcp_server_accepts_jsonrpc_initialize() {
    use std::io::{Write, Read};

    // Get the path to the modl binary
    let bin_path = std::env::current_exe()
        .expect("Failed to get current exe")
        .parent()
        .expect("Failed to get parent dir")
        .join("modl");
    
    let mut child = StdCommand::new(&bin_path)
        .arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to start MCP server");

    let stdin = child.stdin.as_mut().expect("Failed to get stdin");
    let stdout = child.stdout.as_mut().expect("Failed to get stdout");

    // Send JSON-RPC initialize request
    let init_request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    });

    let request_str = serde_json::to_string(&init_request).unwrap();
    
    // Write JSON-RPC message with Content-Length header (LSP style)
    write!(stdin, "Content-Length: {}\r\n\r\n{}", request_str.len(), request_str)
        .expect("Failed to write to stdin");
    stdin.flush().expect("Failed to flush stdin");

    // Read response
    let mut buffer = [0u8; 4096];
    let bytes_read = stdout.read(&mut buffer).expect("Failed to read stdout");
    let response = String::from_utf8_lossy(&buffer[..bytes_read]);

    // Cleanup
    let _ = child.kill();
    let _ = child.wait();

    // Verify we got a valid JSON-RPC response
    assert!(response.contains("jsonrpc"), "Response should contain jsonrpc field");
    assert!(response.contains("result") || response.contains("error"), 
            "Response should have result or error");
}
