from typing import TypedDict, Sequence, Annotated, List, Literal
from langgraph.graph.message import BaseMessage, add_messages
import operator

class CodeConvertState(TypedDict):
    """State for code conversion workflow"""
    original_code: str
    generation: List[str]
    reflection: List[str]
    judgement: Literal['PASS', 'FAIL']
    iterations: int

def manage_memory(state: CodeConvertState) -> CodeConvertState:
    """Manage memory by adding last 2 items in generation and reflection"""
    if len(state["generation"]) > 2:
        state["generation"] = state["generation"][-2:]
    
    if len(state["reflection"]) > 2:
        state["reflection"] = state["reflection"][-2:]

    return state