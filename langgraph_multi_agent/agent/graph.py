from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from agent.state import State
from agent.nodes import supervisor_node, travel_node, joke_node, couplet_node, other_node
from agent.router import routing_func

# 构建图
builder = StateGraph(State)

# 节点
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("travel_node", travel_node)
builder.add_node("joke_node", joke_node)
builder.add_node("couplet_node", couplet_node)
builder.add_node("other_node", other_node)

# 边
builder.add_edge(START, "supervisor_node")
builder.add_conditional_edges("supervisor_node", routing_func)
builder.add_edge("travel_node", END)
builder.add_edge("joke_node", END)
builder.add_edge("couplet_node", END)
builder.add_edge("other_node", END)

# 内存记忆
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)