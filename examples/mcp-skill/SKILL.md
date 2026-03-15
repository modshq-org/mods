# modl-mcp

MCP (Model Context Protocol) integration for modl image generation.

## Description

This skill enables AI assistants to generate images using modl through the Model Context Protocol (MCP). It provides a bridge between natural language requests and modl's image generation capabilities.

## When to Use

Use this skill when:
- The user wants to generate images using AI models
- The user asks about available models in modl
- The user wants to check GPU/training status
- The user needs to download new models

## Tools

### modl_generate

Generate an image using modl's AI models.

**Parameters:**
- `prompt` (required): Text description of the image to generate
- `base`: Base model to use (e.g., "flux-dev", "flux-schnell", "sdxl-base-1.0")
- `lora`: LoRA name or path to apply
- `lora_strength`: LoRA weight (0.0-1.0)
- `size`: Image size preset ("1:1", "16:9", "9:16", "4:3", "3:4")
- `steps`: Number of inference steps
- `guidance`: Guidance scale
- `seed`: Random seed for reproducibility
- `count`: Number of images to generate

**Example:**
```json
{
  "prompt": "a cat astronaut on mars",
  "base": "flux-dev",
  "size": "16:9",
  "steps": 30
}
```

### modl_list_models

List all installed models in modl.

**Parameters:**
- `asset_type`: Optional filter by type ("checkpoint", "lora", "vae")

### modl_pull_model

Download a model from the registry or HuggingFace.

**Parameters:**
- `model_id` (required): Model ID (e.g., "flux-dev", "hf:owner/model")
- `variant`: Force specific variant ("fp16", "fp8", "gguf-q4")

### modl_get_status

Get system status including GPU info and training runs.

**Parameters:**
- `run_name`: Optional specific training run name

## Prerequisites

1. modl must be installed: https://modl.run
2. `modl serve` must be running
3. MCP client must be configured

## Configuration

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "modl": {
      "command": "modl",
      "args": ["mcp"]
    }
  }
}
```

### Cursor

Add to Cursor MCP settings with command `modl mcp`.

## Error Handling

The skill handles common errors:
- modl serve not running → Prompt user to start it
- Model not found → Suggest available models
- GPU out of memory → Suggest smaller model or quantized variant

## Examples

**Simple generation:**
```
User: "Generate a picture of a sunset"
→ Calls modl_generate with prompt "a sunset"
```

**Advanced generation:**
```
User: "Create a 16:9 image of a cyberpunk city using flux-dev with 50 steps"
→ Calls modl_generate with all parameters
```

**Model management:**
```
User: "What models do I have?"
→ Calls modl_list_models

User: "Download flux-schnell"
→ Calls modl_pull_model with model_id "flux-schnell"
```

## Notes

- Image generation is asynchronous; results appear in modl's output folder
- Default model is flux-schnell if not specified
- The skill automatically checks if modl serve is running before making requests
