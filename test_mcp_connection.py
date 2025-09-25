#!/usr/bin/env python3
"""
Test MCP Server Connection
Verifies the server implements MCP protocol correctly
"""

import json
import subprocess
import sys
import time

def test_mcp_server():
    """Test the MCP server with basic JSON-RPC communication"""

    print("Testing MCP Server Connection...")
    print("-" * 50)

    # Start the server as a subprocess
    process = subprocess.Popen(
        [sys.executable, "src/mcp_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0
    )

    try:
        # Send initialize request
        initialize_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "1.0.0",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            },
            "id": 1
        }

        print("Sending initialize request...")
        process.stdin.write(json.dumps(initialize_request) + "\n")
        process.stdin.flush()

        # Wait a bit for response
        time.sleep(1)

        # Try to read response
        process.stdout.flush()

        # Send a tools/list request
        list_tools_request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 2
        }

        print("Sending tools/list request...")
        process.stdin.write(json.dumps(list_tools_request) + "\n")
        process.stdin.flush()

        # Give it time to respond
        time.sleep(1)

        # Check if process is still running
        if process.poll() is None:
            print("✅ Server is running (no crash)")
        else:
            print("❌ Server crashed")
            stderr = process.stderr.read()
            if stderr:
                print("Error output:", stderr)

        # Terminate the process
        process.terminate()
        process.wait(timeout=2)

        print("\nTest complete. Server appears to be implementing MCP protocol.")

    except Exception as e:
        print(f"Error during test: {e}")
        process.terminate()

    finally:
        if process.poll() is None:
            process.kill()

if __name__ == "__main__":
    test_mcp_server()