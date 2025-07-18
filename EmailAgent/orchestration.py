import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from email_extractor import EmailExtractor
from email_responder import EmailResponder
from utils import make_agent_node

load_dotenv()

#Set location of extracted emails
emails = os.getenv("EMAILS", "data/emails.txt")
extracts = os.getenv("EXTRACTS", "data/extract.json")

#Initialize agents
reader = EmailExtractor(emails)
responder = EmailResponder(extracts)

#Create nodes for the agents
nodes = {
    "read_email": make_agent_node(reader),
    "respond": make_agent_node(responder),
}

class AgentState(TypedDict):
    emails: List[str]
    responses: List[str]

builder = StateGraph(state_schema=AgentState)

# Add nodes
builder.add_node("read_email", nodes["read_email"])
builder.add_node("respond", nodes["respond"])

# Set edges (order of execution)
builder.set_entry_point("read_email")
builder.add_edge("read_email", "respond")
builder.set_finish_point("respond")

# Build the graph
workflow = builder.compile()

# Run the workflow
initial_state = {"emails": [], "responses": []}
final_state = workflow.invoke(initial_state)

print(final_state)