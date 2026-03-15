# modl-mcp

MCP (Model Context Protocol) skill for modl image generation.

This skill provides easy integration with modl's MCP server for AI image generation.

## Installation

1. Ensure modl is installed and available in PATH:
   ```bash
   curl -fsSL https://modl.run/install.sh | sh
   ```

2. Configure the skill in your OpenClaw config:
   ```yaml
   skills:
     - name: modl-mcp
       path: /path/to/modl-mcp
   ```

3. Start modl serve:
   ```bash
   modl serve
   ```

## Usage

The skill provides the following tools:

- `modl_generate` - Generate images from text prompts
- `modl_list_models` - List available models
- `modl_pull_model` - Download new models
- `modl_get_status` - Check GPU and training status

## Example

```
Generate an image of a cat on mars using flux-dev
```

The skill will automatically:
1. Check if modl serve is running
2. Call the appropriate MCP tool
3. Return the result with output location

## Requirements

- modl CLI installed and in PATH
- modl serve running (typically on port 3333)
- MCP client configured (Claude Desktop, Cursor, etc.)

## Configuration

Add to your MCP client config:

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

For Claude Desktop: `~/Library/Application Support/Claude/claude_desktop_config.json`

## License

MIT - Same as modl
