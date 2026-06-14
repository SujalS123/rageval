# MCP Server (Model Context Protocol)

`rageval` ships with a built-in Model Context Protocol (MCP) server. This allows AI coding assistants like Claude Desktop, Cursor, and Windsurf to automatically evaluate the RAG pipelines they build for you.

When enabled, your AI assistant can:
1. Run `rageval_evaluate` directly from the chat interface.
2. See exactly which sentences hallucinated.
3. Automatically rewrite the retriever or prompt based on the failure evidence.

## Installation

```bash
pip install rageval-core[mcp]
```

## Configuration

Add the `rageval` server to your assistant's configuration file.

### Cursor

Add this to Cursor's MCP configuration settings (Cursor Settings > Features > MCP):

* **Name:** `rageval`
* **Type:** `command`
* **Command:** `python -m rageval.mcp_server`

### Claude Desktop

Edit your `claude_desktop_config.json`:
* Mac: `~/Library/Application Support/Claude/claude_desktop_config.json`
* Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "rageval": {
      "command": "python",
      "args": ["-m", "rageval.mcp_server"],
      "env": {
        "ANTHROPIC_API_KEY": "your-api-key"
      }
    }
  }
}
```

### Windsurf

Add this to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "rageval": {
      "command": "python",
      "args": ["-m", "rageval.mcp_server"]
    }
  }
}
```

## Available Tools

Once connected, your AI assistant will have access to:

* `rageval_evaluate`: Runs full evaluation (Faithfulness, ContextPrecision, AnswerRelevancy) on a provided sample and returns the exact hallucination evidence.
* `rageval_batch`: Runs evaluation across multiple samples concurrently.
* `rageval_get_history`: Reads the local SQLite database to analyze historical regressions.
