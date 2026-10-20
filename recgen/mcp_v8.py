"""v8's tool server for one agent reading: a minimal MCP server over stdio that forwards each tool
call to the backend (recgen/probe_v8.Backend) with this reading's handle.

Run by the claude CLI as `python mcp_v8.py <backend url> <handle>`. It uses the standard library
only, so it starts fast, and it never sees which unit it serves: only the backend knows.
"""

import json
import sys
import urllib.request


def backend(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def respond(message: dict, url: str, handle: str) -> dict | None:
    """The JSON-RPC reply to one message, or None for a notification."""
    if "id" not in message:
        return None
    method, params = message.get("method"), message.get("params") or {}
    if method == "initialize":
        result = {"protocolVersion": params.get("protocolVersion"), "capabilities": {"tools": {}},
                  "serverInfo": {"name": "probe", "version": "1"}}
    elif method == "tools/list":
        result = backend(url)
    elif method == "tools/call":
        text = backend(url, {"handle": handle, "tool": params.get("name"), "arguments": params.get("arguments")})["text"]
        result = {"content": [{"type": "text", "text": text}]}
    elif method == "ping":
        result = {}
    else:
        return {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": f"unsupported: {method}"}}
    return {"jsonrpc": "2.0", "id": message["id"], "result": result}


def main() -> None:
    url, handle = sys.argv[1], sys.argv[2]
    for line in sys.stdin:
        if line.strip():
            reply = respond(json.loads(line), url, handle)
            if reply is not None:
                sys.stdout.write(json.dumps(reply) + "\n")
                sys.stdout.flush()


if __name__ == "__main__":
    main()
