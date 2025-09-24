# Changelog

All notable changes to the Converse-Enhanced MCP project will be documented in this file.

## [1.1.0] - 2025-09-25

### Fixed
- **Ollama-first routing now properly enforced** - Fixed issue where Ollama was not being selected even when available
- **Removed artificial rate limiting on Ollama** - Local Ollama requests are never rate-limited
- **Fixed initial status configuration** - Ollama now starts with AVAILABLE status when detected
- **Improved provider selection logic** - Auto mode now always checks Ollama first

### Added
- **Intelligent model selection** - Automatically selects optimal Ollama model based on query complexity:
  - Simple queries (<50 chars) → phi3:mini (fastest)
  - Code-related queries → codellama:7b
  - Complex queries (>200 chars) → qwen2.5-coder:32b
  - Default queries → llama3.2:3b

### Changed
- Priority system now strictly enforces FREE tier first (Ollama priority = 1)
- Auto model selection optimized for query complexity
- Provider status handling improved to prevent false rate limiting

### Performance
- **100% FREE tier routing** when Ollama is available
- Response times: 1-15 seconds based on model size
- No unexpected API costs when local models are available

## [1.0.0] - 2025-09-24

### Initial Release
- MCP protocol implementation for Claude Desktop
- Ollama auto-detection for local models
- Support for multiple AI providers (OpenAI, Anthropic, Google, XAI, Perplexity)
- Cost optimization with Ollama prioritization
- Usage statistics tracking