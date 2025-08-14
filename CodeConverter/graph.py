from typing import Dict, Any
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from states import CodeConvertState
from agents import CodeConverterAgent, ReflectionAgent

class CodeConversionWorkflow:
    """Workflow for converting code using agents."""
    
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.converter_agent = CodeConverterAgent()
        self.reflection_agent = ReflectionAgent()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the state graph for the code conversion workflow."""
        graph = StateGraph(CodeConvertState)
        
        # Define nodes and their corresponding functions
        graph.add_node("converter", self._converter_node)
        graph.add_node("reflector", self._reflector_node)
        graph.add_node("revisor", self._revisor_node)

        # Set entry point
        graph.set_entry_point("converter")

        # Define edges between nodes
        graph.add_edge("converter", "reflector")
        graph.add_edge("revisor", "reflector")
        
        #Add conditional edge
        graph.add_conditional_edges(
            "reflector",
           self._should_continue,
            {
                "continue": "revisor",
                "end": END
            }
        )

        #app = graph.compile()
        #png_bytes = app.get_graph(xray=True).draw_mermaid_png()
        # # Save to a local file
        #with open("workflow_diagram.png", "wb") as f:
        #    f.write(png_bytes)

        memory = MemorySaver()
        return graph.compile(checkpointer=memory)
    
    def _converter_node(self, state: CodeConvertState) -> CodeConvertState:
        """Node for initial code conversion."""
        return self.converter_agent.convert_code(state)
    
    def _reflector_node(self, state: CodeConvertState) -> CodeConvertState:
        """Node for reflecting on the code."""
        return self.reflection_agent.reflect_on_code(state)
    
    def _revisor_node(self, state: CodeConvertState) -> CodeConvertState:
        """Node for revising the code based on feedback."""
        return self.converter_agent.convert_code(state)
        
    def _should_continue(self, state: CodeConvertState) -> str:
        """Determine if the workflow should continue based on iterations and reflection"""

        #Feedback check (based on heuristics)
        if state["judgement"].upper() == "PASS":
            return "end"
        
        #Max iterations check
        if state["iterations"] >= self.max_iterations:
            return "end"
            
        return "continue"


    def run(self, original_code: str, thread_id: str = "default") -> Dict[str, Any]:
        """Run the code conversion workflow."""
        initial_state = CodeConvertState(
                                        original_code=original_code,
                                        generation=[],
                                        reflection=[],
                                        judgement='FAIL',
                                        iterations=0
                                    )
        
        config = {"configurable": {"thread_id": thread_id}}

        # Invoke the graph with the initial state
        result = self.graph.invoke(initial_state, config=config)
        return result
    
    def stream_workflow(self, original_code: str, thread_id: str = "default"):
        """Stream the workflow for real-time updates."""
        initial_state = CodeConvertState(
                                        original_code=original_code,
                                        generation=[],
                                        reflection=[],
                                        judgement='FAIL',
                                        iterations=0
                                    )
        
        config = {"configurable": {"thread_id": thread_id}}

        # Stream the graph with the initial state
        for event in self.graph.stream(initial_state, config=config):
            yield event
