from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Demo")

#Multiply tool
@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers together."""
    return a * b

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Returns a greeting message."""
    return f"Hello, {name}!"

@mcp.prompt()
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"
