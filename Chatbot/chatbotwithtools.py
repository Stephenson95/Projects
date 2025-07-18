from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain.agents import tool
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_tavily import TavilySearch
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import pprintpp

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

#Define the tools
weather_tool = OpenWeatherMapAPIWrapper()
search_tool = TavilySearch(max_results=3)
tools = [search_tool, weather_tool.run]

llm_with_tools = llm.bind_tools(tools=tools)

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: ChatState) -> StateGraph:
    return {
        "messages": [llm_with_tools.invoke(state["messages"])]
    }


def tools_router(state: ChatState):
    last_message = state["messages"][-1]

    if(hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tool_node"
    else: 
        return END

tool_node = ToolNode(tools = tools)

graph = StateGraph(ChatState)

graph.add_node("chatbot", chatbot)
graph.add_node("tool_node", tool_node)
graph.set_entry_point("chatbot")

graph.add_conditional_edges("chatbot", tools_router)
graph.add_edge("tool_node", "chatbot")

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

# def print_stream(stream):
#     for s in stream:
#         message = s["messages"][-1]
#         if isinstance(message, tuple):
#             print(message)
#         else:
#             print(message.pretty_print())

# inputs = {"messages": [HumanMessage(content="What is the weather in Sydney Carlingford?")]}
# print_stream(app.stream(inputs, stream_mode="values"))