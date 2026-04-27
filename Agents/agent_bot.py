from typing import TypedDict, List
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    messages: List[BaseMessage]


llm = ChatOllama(
    model="llama3"
)

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])

    print(f"\nAI: {response.content}")

    # Store conversation history
    state["messages"].append(response)

    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)

graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

messages = []

while True:
    user_input = input("Enter: ")

    if user_input.lower() == "exit":
        break

    messages.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": messages})

    messages = result["messages"]