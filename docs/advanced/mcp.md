# MCP Server Setup

rageval includes a Model Context Protocol (MCP) server. This allows AI coding assistants like Claude Desktop, Cursor, and Windsurf to directly run RAG evaluations, check history, and debug failing pipelines for you.

## Installation

Ensure you have installed the `mcp` extra:

```bash
pip install rageval[mcp]
```

## Available Tools

Once configured, your AI assistant will have access to:
- `evaluate_sample` — Run metrics on a specific query/docs/answer combo.
- `batch_evaluate` — Run metrics across a JSON dataset.
- `get_run_history` — Check if your pipeline degraded recently.
- `analyze_consistency` — Check if paraphrases break the pipeline.

## Claude Desktop Configuration

Edit your `claude_desktop_config.json` (usually at `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS or `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "rageval": {
      "command": "rageval",
      "args": ["serve"],
      "env": {
        "ANTHROPIC_API_KEY": "your-api-key"
      }
    }
  }
}
```

## Cursor Configuration

1. Open Cursor Settings (Cmd/Ctrl + ,)
2. Navigate to **Features** > **MCP Servers**
3. Click **+ Add new MCP server**
4. Configure as follows:
   - **Name**: rageval
   - **Type**: command
   - **Command**: `rageval serve`
5. Make sure your environment variables (like `OPENAI_API_KEY`) are set in your terminal or `.env` file where Cursor runs.

## Windsurf Configuration

Edit your `mcp_config.json` (usually at `~/.codeium/windsurf/mcp_config.json`):

```json
{
  "mcpServers": {
    "rageval": {
      "command": "rageval",
      "args": ["serve"],
      "env": {
        "OPENAI_API_KEY": "your-api-key"
      }
    }
  }
}
```

Once configured, try asking your assistant: "Run a rageval evaluation on `eval_data.json` and tell me which metrics are failing."
