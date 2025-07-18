from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: ChatState) -> StateGraph:
    return {
        "messages": [llm.invoke(state["messages"])]
    }

graph = StateGraph(ChatState)

graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.set_finish_point("chatbot", END)

app = graph.compile()

while True:
    user_input = input("User: ")
    if (user_input.lower() in ["exit", "quit"]):
        break
    else:
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        })

        print(f'Chatbot: {result["messages"][-1].content}')