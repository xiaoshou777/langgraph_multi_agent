from langgraph.graph import END
from agent.state import State

def routing_func(state: State):
    t = state["type"]
    if t == "travel": return "travel_node"
    if t == "joke": return "joke_node"
    if t == "couplet": return "couplet_node"
    if t == "other": return "other_node"
    if t == "done": return END
    return "other_node"