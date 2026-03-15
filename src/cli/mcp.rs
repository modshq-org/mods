//! MCP (Model Context Protocol) server for modl.
//!
//! This module implements an MCP server that exposes modl's image generation
//! and model management capabilities to MCP clients like Claude Desktop,
//! Cursor, and other AI assistants.
//!
//! Uses JSON-RPC over stdio as per MCP specification.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

/// MCP protocol version
const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// MCP server implementation for modl
pub struct McpServer {
    /// Base URL for the modl API (e.g., http://127.0.0.1:3333)
    api_base: String,
}

impl McpServer {
    /// Create a new MCP server.
    pub fn new(api_base: String) -> Self {
        Self { api_base }
    }

    /// Create a new MCP server with default localhost API.
    #[allow(dead_code)]
    pub fn new_localhost() -> Self {
        Self::new("http://127.0.0.1:3333".to_string())
    }

    /// Run the MCP server over stdio.
    pub async fn run(&self) -> Result<()> {
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        let reader = stdin.lock();

        for line in reader.lines() {
            let line = line.context("Failed to read line from stdin")?;
            if line.trim().is_empty() {
                continue;
            }

            match self.handle_request(&line).await {
                Some(response) => {
                    let response_json = serde_json::to_string(&response)?;
                    writeln!(stdout, "{}", response_json)?;
                    stdout.flush()?;
                }
                None => {
                    // Notification - no response needed
                }
            }
        }

        Ok(())
    }

    /// Handle a single JSON-RPC request.
    async fn handle_request(&self,
        request_str: &str,
    ) -> Option<JsonRpcResponse> {
        let request: JsonRpcRequest = match serde_json::from_str(request_str) {
            Ok(req) => req,
            Err(e) => {
                return Some(JsonRpcResponse::error(
                    None,
                    -32700,
                    format!("Parse error: {}", e),
                ));
            }
        };

        let id = request.id.clone();

        // Handle notifications (no id)
        if id.is_none() {
            // Process notification but don't return response
            let _ = self.handle_method(&request.method, request.params).await;
            return None;
        }

        match self.handle_method(&request.method, request.params).await {
            Ok(result) => Some(JsonRpcResponse::success(id, result)),
            Err((code, message)) => Some(JsonRpcResponse::error(id, code, message)),
        }
    }

    /// Handle a method call.
    async fn handle_method(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> Result<Value, (i32, String)> {
        match method {
            "initialize" => self.handle_initialize(params).await,
            "initialized" => {
                // Notification, no response needed
                Ok(json!({}))
            }
            "tools/list" => self.handle_tools_list().await,
            "tools/call" => self.handle_tools_call(params).await,
            _ => Err((-32601, format!("Method not found: {}", method))),
        }
    }

    /// Handle initialize request.
    async fn handle_initialize(
        &self,
        _params: Option<Value>,
    ) -> Result<Value, (i32, String)> {
        Ok(json!({
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "modl-mcp-server",
                "version": env!("CARGO_PKG_VERSION"),
            }
        }))
    }

    /// Handle tools/list request.
    async fn handle_tools_list(&self,
    ) -> Result<Value, (i32, String)> {
        Ok(json!({
            "tools": [
                {
                    "name": "modl_generate",
                    "description": "Generate an image using AI models",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The text prompt describing the image to generate"
                            },
                            "base": {
                                "type": "string",
                                "description": "Base model to use (e.g., flux-dev, flux-schnell, sdxl-base-1.0)"
                            },
                            "lora": {
                                "type": "string",
                                "description": "LoRA name or path to apply"
                            },
                            "lora_strength": {
                                "type": "number",
                                "description": "LoRA strength/weight (0.0 = no effect, 1.0 = full strength)"
                            },
                            "size": {
                                "type": "string",
                                "description": "Image size preset: 1:1, 16:9, 9:16, 4:3, 3:4 or WxH format"
                            },
                            "steps": {
                                "type": "integer",
                                "description": "Number of inference steps"
                            },
                            "guidance": {
                                "type": "number",
                                "description": "Guidance scale"
                            },
                            "seed": {
                                "type": "integer",
                                "description": "Random seed for reproducibility"
                            },
                            "count": {
                                "type": "integer",
                                "description": "Number of images to generate"
                            }
                        },
                        "required": ["prompt"]
                    }
                },
                {
                    "name": "modl_list_models",
                    "description": "List installed models and their details",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "asset_type": {
                                "type": "string",
                                "description": "Filter by asset type (e.g., checkpoint, lora, vae)"
                            }
                        }
                    }
                },
                {
                    "name": "modl_pull_model",
                    "description": "Download a model from the registry or HuggingFace",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "model_id": {
                                "type": "string",
                                "description": "Model ID to pull (e.g., flux-dev, hf:stabilityai/stable-diffusion-xl-base-1.0)"
                            },
                            "variant": {
                                "type": "string",
                                "description": "Force a specific variant (e.g., fp16, fp8, gguf-q4)"
                            }
                        },
                        "required": ["model_id"]
                    }
                },
                {
                    "name": "modl_get_status",
                    "description": "Get system status including GPU info and training runs",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "run_name": {
                                "type": "string",
                                "description": "Optional run name to get specific training status"
                            }
                        }
                    }
                }
            ]
        }))
    }

    /// Handle tools/call request.
    async fn handle_tools_call(
        &self,
        params: Option<Value>,
    ) -> Result<Value, (i32, String)> {
        let params = params.ok_or((-32602, "Missing params".to_string()))?;
        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or((-32602, "Missing tool name".to_string()))?;
        let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

        match name {
            "modl_generate" => self.tool_generate(arguments).await,
            "modl_list_models" => self.tool_list_models(arguments).await,
            "modl_pull_model" => self.tool_pull_model(arguments).await,
            "modl_get_status" => self.tool_get_status(arguments).await,
            _ => Err((-32601, format!("Tool not found: {}", name))),
        }
    }

    /// modl_generate tool implementation.
    async fn tool_generate(&self, args: Value) -> Result<Value, (i32, String)> {
        let prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .ok_or((-32602, "Missing required parameter: prompt".to_string()))?;

        let client = reqwest::Client::new();

        let body = json!({
            "prompt": prompt,
            "count": args.get("count").and_then(|v| v.as_u64()).unwrap_or(1),
            "base_model": args.get("base"),
            "lora": args.get("lora"),
            "lora_strength": args.get("lora_strength"),
            "size": args.get("size"),
            "steps": args.get("steps"),
            "guidance": args.get("guidance"),
            "seed": args.get("seed"),
        });

        match client
            .post(format!("{}/api/generate", self.api_base))
            .json(&body)
            .send()
            .await
        {
            Ok(response) => {
                if response.status().is_success() {
                    match response.json::<Value>().await {
                        Ok(result) => {
                            let output_id = result
                                .get("id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown");
                            Ok(json!({
                                "content": [
                                    {
                                        "type": "text",
                                        "text": format!(
                                            "Image generation started. Output ID: {}\n\nView results at: {}/files/outputs/",
                                            output_id, self.api_base
                                        )
                                    }
                                ]
                            }))
                        }
                        Err(e) => Err((-32603, format!("Failed to parse response: {}", e))),
                    }
                } else {
                    let text = response.text().await.unwrap_or_default();
                    Err((-32603, format!("API error: {}", text)))
                }
            }
            Err(e) => Err((-32603, format!("Request failed: {}", e))),
        }
    }

    /// modl_list_models tool implementation.
    async fn tool_list_models(&self,
        _args: Value,
    ) -> Result<Value, (i32, String)> {
        let client = reqwest::Client::new();

        match client
            .get(format!("{}/api/models", self.api_base))
            .send()
            .await
        {
            Ok(response) => {
                if response.status().is_success() {
                    match response.json::<Vec<Value>>().await {
                        Ok(models) => {
                            let mut message = String::from("Installed models:\n\n");
                            for model in models {
                                let id = model
                                    .get("id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("unknown");
                                let params = model
                                    .get("params_b")
                                    .and_then(|v| v.as_f64())
                                    .map(|p| format!("{:.1}B", p))
                                    .unwrap_or_else(|| "?".to_string());
                                message.push_str(&format!("- {} ({} params)\n", id, params));
                            }
                            Ok(json!({
                                "content": [{"type": "text", "text": message}]
                            }))
                        }
                        Err(e) => Err((-32603, format!("Failed to parse models: {}", e))),
                    }
                } else {
                    Err((-32603, "Failed to fetch models".to_string()))
                }
            }
            Err(e) => Err((-32603, format!("Request failed: {}", e))),
        }
    }

    /// modl_pull_model tool implementation.
    async fn tool_pull_model(&self,
        args: Value,
    ) -> Result<Value, (i32, String)> {
        let model_id = args
            .get("model_id")
            .and_then(|v| v.as_str())
            .ok_or((-32602, "Missing required parameter: model_id".to_string()))?;

        let client = reqwest::Client::new();

        let body = json!({
            "id": model_id,
            "variant": args.get("variant"),
        });

        match client
            .post(format!("{}/api/models/install", self.api_base))
            .json(&body)
            .send()
            .await
        {
            Ok(response) => {
                if response.status().is_success() {
                    Ok(json!({
                        "content": [{
                            "type": "text",
                            "text": format!(
                                "Model '{}' is being downloaded. This may take several minutes.",
                                model_id
                            )
                        }]
                    }))
                } else {
                    let text = response.text().await.unwrap_or_default();
                    Err((-32603, format!("Failed to pull model: {}", text)))
                }
            }
            Err(e) => Err((-32603, format!("Request failed: {}", e))),
        }
    }

    /// modl_get_status tool implementation.
    async fn tool_get_status(&self,
        args: Value,
    ) -> Result<Value, (i32, String)> {
        let client = reqwest::Client::new();
        let mut message_parts = Vec::new();

        // Get GPU status
        match client
            .get(format!("{}/api/gpu", self.api_base))
            .send()
            .await
        {
            Ok(response) => {
                if let Ok(gpu) = response.json::<Value>().await {
                    if let Some(name) = gpu.get("name").and_then(|v| v.as_str()) {
                        let vram = gpu
                            .get("vram_mb")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        let used = gpu
                            .get("vram_used_mb")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        message_parts.push(format!(
                            "GPU: {} ({} MB / {} MB used)",
                            name, used, vram
                        ));
                    }
                }
            }
            Err(_) => {}
        }

        // Get training runs
        let runs_url = if let Some(run_name) = args.get("run_name").and_then(|v| v.as_str()) {
            format!("{}/api/status/{}", self.api_base, run_name)
        } else {
            format!("{}/api/status", self.api_base)
        };

        match client.get(&runs_url).send().await {
            Ok(response) => {
                if let Ok(runs) = response.json::<Vec<Value>>().await {
                    if !runs.is_empty() {
                        message_parts.push("\nTraining runs:".to_string());
                        for run in runs {
                            let name = run
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown");
                            let status = if run
                                .get("is_running")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false)
                            {
                                "running"
                            } else {
                                "stopped"
                            };
                            let progress = run
                                .get("progress")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.0);
                            message_parts.push(format!(
                                "- {}: {} ({:.1}%)",
                                name,
                                status,
                                progress * 100.0
                            ));
                        }
                    }
                }
            }
            Err(_) => {}
        }

        let message = if message_parts.is_empty() {
            "No status information available.".to_string()
        } else {
            message_parts.join("\n")
        };

        Ok(json!({
            "content": [{"type": "text", "text": message}]
        }))
    }
}

/// JSON-RPC request structure.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct JsonRpcRequest {
    jsonrpc: String,
    method: String,
    #[serde(default)]
    params: Option<Value>,
    #[serde(default)]
    id: Option<Value>,
}

/// JSON-RPC response structure.
#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

/// JSON-RPC error structure.
#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

impl JsonRpcResponse {
    /// Create a success response.
    fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Create an error response.
    fn error(id: Option<Value>, code: i32, message: String) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message,
                data: None,
            }),
        }
    }
}

/// Run the MCP server.
pub async fn run(port: Option<u16>) -> Result<()> {
    let api_base = if let Some(port) = port {
        format!("http://127.0.0.1:{}", port)
    } else {
        "http://127.0.0.1:3333".to_string()
    };

    let server = McpServer::new(api_base);

    eprintln!("Starting modl MCP server...");
    eprintln!("Connecting to modl API at: {}", server.api_base);

    // Check if API is available
    let client = reqwest::Client::new();
    match client
        .get(format!("{}/api/models", server.api_base))
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            eprintln!("✓ modl API is available");
        }
        _ => {
            eprintln!("⚠ Warning: modl API is not available. Make sure 'modl serve' is running.");
        }
    }

    server.run().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jsonrpc_response_success() {
        let response = JsonRpcResponse::success(Some(json!(1)), json!({"result": "ok"}));
        let json_str = serde_json::to_string(&response).unwrap();
        assert!(json_str.contains("\"jsonrpc\":\"2.0\""));
        assert!(json_str.contains("\"result\""));
    }

    #[test]
    fn test_jsonrpc_response_error() {
        let response = JsonRpcResponse::error(Some(json!(1)), -32600, "Invalid request".to_string());
        let json_str = serde_json::to_string(&response).unwrap();
        assert!(json_str.contains("\"jsonrpc\":\"2.0\""));
        assert!(json_str.contains("\"error\""));
        assert!(json_str.contains("-32600"));
    }

    #[tokio::test]
    async fn test_mcp_server_new() {
        let server = McpServer::new_localhost();
        assert_eq!(server.api_base, "http://127.0.0.1:3333");
    }

    #[tokio::test]
    async fn test_mcp_server_custom_port() {
        let server = McpServer::new("http://127.0.0.1:8080".to_string());
        assert_eq!(server.api_base, "http://127.0.0.1:8080");
    }
}
