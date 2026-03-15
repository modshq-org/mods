# MCP Examples

This directory contains examples and integrations for modl's MCP (Model Context Protocol) server.

## Contents

### mcp-skill/

An OpenClaw skill for easy MCP integration. Provides:
- Documentation for setting up MCP with various clients
- Helper script to check modl installation status
- Configuration examples for Claude Desktop, Cursor, and other MCP clients

## Quick Start

1. Start modl serve:
   ```bash
   modl serve
   ```

2. Configure your MCP client (see mcp-skill/README.md)

3. Use natural language to generate images:
   - "Generate a picture of a cat on mars"
   - "Create a 16:9 cyberpunk cityscape using flux-dev"
   - "List my installed models"

## Available Tools

- `modl_generate` - Generate images from text prompts
- `modl_list_models` - List installed models
- `modl_pull_model` - Download new models
- `modl_get_status` - Check GPU and training status

## More Information

See the main MCP documentation in the project README.
