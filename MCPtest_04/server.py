from starlette.applications import Starlette
from starlette.routing import Mount, Host
from mcp.server.fastmcp import FastMCP

mcp = FastMCP()

app = Starlette(
    routes=[
                Mount('/', app=mcp.sse_app())
            ]
)

@mcp.tool()
def add(a: int, b:int) -> int:
    """Add two integers together"""
    return a + b
