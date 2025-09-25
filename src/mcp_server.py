#!/usr/bin/env python3
"""
MCP Server for Converse-Enhanced with Ollama Auto-Detection
Implements the Model Context Protocol for Claude Desktop
"""

import asyncio
import json
import os
import sys
import logging
from typing import Any, Sequence
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import our API manager and Ollama manager
from server import OptimizedAPIManager
from ollama_manager import OllamaManager

# Import MCP SDK
try:
    import mcp.types as types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.server.models import InitializationOptions
except ImportError:
    print("ERROR: MCP SDK not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("converse-enhanced-mcp")

# Create server instance
server = Server("converse-enhanced")

# Global API manager instance
api_manager = None


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="chat",
            description="Send a message to an AI model (Ollama prioritized)",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The message to send to the AI"
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional: specific model to use (e.g., 'llama3.2', 'codellama', 'gpt-3.5-turbo')"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Optional: temperature for response randomness (0.0-1.0)"
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Optional: maximum tokens in response"
                    }
                },
                "required": ["prompt"]
            }
        ),
        types.Tool(
            name="list_models",
            description="List all available AI models (Ollama and API models)",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        types.Tool(
            name="get_status",
            description="Get status of all AI providers and usage statistics",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        types.Tool(
            name="refresh_ollama",
            description="Refresh the list of available Ollama models",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> Sequence[types.TextContent]:
    """Handle tool calls"""
    global api_manager

    # Ensure API manager is initialized
    if api_manager is None:
        api_manager = OptimizedAPIManager()
        logger.info(f"Initialized API manager with {len(api_manager.providers)} providers")

    try:
        if name == "chat":
            # Send message to AI
            prompt = arguments.get("prompt", "")
            model = arguments.get("model")
            temperature = arguments.get("temperature")
            max_tokens = arguments.get("max_tokens")

            if not prompt:
                return [types.TextContent(
                    type="text",
                    text="Error: Prompt is required"
                )]

            # Prepare kwargs
            kwargs = {}
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            try:
                response, provider_used = await api_manager.send_message(
                    prompt,
                    model=model,
                    **kwargs
                )

                # Include provider info in response
                result = f"[Provider: {provider_used}]\n{response}"

                # Log usage
                logger.info(f"Request completed via {provider_used}")

                return [types.TextContent(
                    type="text",
                    text=result
                )]

            except Exception as e:
                logger.error(f"Error calling AI: {e}")
                return [types.TextContent(
                    type="text",
                    text=f"Error: {str(e)}"
                )]

        elif name == "list_models":
            # List all available models
            models_info = {
                "ollama": {
                    "status": "available" if api_manager.ollama_manager.is_available else "unavailable",
                    "models": api_manager.ollama_manager.available_models
                },
                "api_providers": {}
            }

            # Add API provider models
            for provider_name, provider in api_manager.providers.items():
                if provider_name != "ollama" and provider.enabled:
                    models_info["api_providers"][provider_name] = {
                        "status": provider.status.value,
                        "models": provider.models[:5] if provider.models else []  # Show first 5
                    }

            return [types.TextContent(
                type="text",
                text=json.dumps(models_info, indent=2)
            )]

        elif name == "get_status":
            # Get full status report
            status = api_manager.get_status_report()
            return [types.TextContent(
                type="text",
                text=json.dumps(status, indent=2)
            )]

        elif name == "refresh_ollama":
            # Refresh Ollama models
            previous_count = len(api_manager.ollama_manager.available_models)
            api_manager.ollama_manager.refresh_models()
            new_count = len(api_manager.ollama_manager.available_models)

            # Update the provider's model list
            if "ollama" in api_manager.providers:
                api_manager.providers["ollama"].models = api_manager.ollama_manager.available_models

            result = {
                "previous_count": previous_count,
                "new_count": new_count,
                "models": api_manager.ollama_manager.available_models,
                "status": "refreshed"
            }

            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]

        else:
            return [types.TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]

    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return [types.TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]


async def main():
    """Main entry point for MCP server"""
    global api_manager

    # Initialize API manager early
    logger.info("Initializing Converse-Enhanced MCP Server...")
    api_manager = OptimizedAPIManager()

    # Log initialization status
    ollama_status = "available" if api_manager.ollama_manager.is_available else "unavailable"
    model_count = len(api_manager.ollama_manager.available_models)
    logger.info(f"Ollama status: {ollama_status} with {model_count} models")

    # Get available providers
    available_providers = []
    for name, provider in api_manager.providers.items():
        if provider.status.value == "available":
            available_providers.append(name)

    logger.info(f"Available providers: {', '.join(available_providers)}")

    # Run the stdio server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="converse-enhanced",
                server_version="1.0.0",
                capabilities={}
            )
        )


if __name__ == "__main__":
    asyncio.run(main())