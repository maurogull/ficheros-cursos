import sys
from fastmcp import FastMCP


mcp = FastMCP("LocalTool1")

@mcp.tool()
def mauristica(number: float) -> float:
    """Devuelve el resultado de aplicar la función Maurística al número dado"""
    return number / 2 + 10


print("Inicia server MCP", file=sys.stderr, flush=True)
mcp.run()
