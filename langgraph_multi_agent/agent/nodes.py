import asyncio
from langchain_core.messages import HumanMessage
from langgraph.config import get_stream_writer
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START, END
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
import os

from agent.state import State
from utils.config import get_llm, PDF_PATH, EMBEDDING_MODEL, NODES

llm = get_llm()

nodes = ["travel", "joke", "couplet", "other"]
# ------------------------------
# 问题分类
# ------------------------------
def supervisor_node(state: State):
    print(">>> supervisor_node")
    writer = get_stream_writer()
    writer({"node": ">>> supervisor_node"})

    # 系统提示词
    prompt = """你是一个专业的客服助手，负责对用户的问题进行分类，并将任务分给其他Agent执行。
如果用户的问题是和旅游路线规划相关的，那就返回 travel。
如果用户的问题是希望讲一个笑话，那就返回 joke。
如果用户的问题是与交通知识相关的，那就返回 couplet。
如果是其他的问题，返回 other。
除了这几个选项外，不要返回任何其他的内容。
"""
    user_content = state["messages"][-1].content
    prompts = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content}
    ]

    if state.get("type") == "done":
        writer({"supervisor_step": "任务处理完成"})
        return {"type": END}

    response = llm.invoke(prompts)
    typeRes = response.content.strip()
    writer({"supervisor_step": f"分类结果为：{typeRes}"})

    if typeRes in nodes:
        return {"type": typeRes}
    else:
        raise ValueError(f"分类结果 {typeRes} 不在允许列表中")

# ------------------------------
# 通用回答
# ------------------------------
def other_node(state: State):
    print(">>> other_node")
    writer = get_stream_writer()
    return {
        "messages": [HumanMessage(content="我暂时无法回答这个问题")],
        "type": "other"
    }

# ------------------------------
# 讲笑话
# ------------------------------
def joke_node(state: State):
    print(">>> joke_node")
    writer = get_stream_writer()
    writer({"node": ">>>> joke_node"})

    system_prompt = "你是一个笑话大师，根据用户的问题，写一个不超过100个字的笑话。"
    user_content = state["messages"][-1].content
    prompts = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    response = llm.invoke(prompts)
    writer({"joke_result": response.content})
    return {"messages": [HumanMessage(content=response.content)], "type": "done"}

# ------------------------------
# 旅游规划（异步 + MCP）
# ------------------------------
async def travel_node(state: State):
    print(">>> travel_node")
    writer = get_stream_writer()
    writer({"node": ">>>> travel_node"})

    # 初始化 MCP 客户端
    client = MultiServerMCPClient({
        "amap-maps-streamableHTTP": {
            "url": "https://mcp.amap.com/mcp?key=xxxxxxxxxx",
            "transport": "streamable_http"
        }
    })

    tools = await client.get_tools()

    # 创建工具调用智能体
    agent = create_react_agent(model=llm, tools=tools)

    # 异步调用智能体
    response = await agent.ainvoke({
        "messages": [
            ("system", "你是一个旅行规划助手，根据用户的问题，生成不超过100个字的自驾路线规划。"),
            ("user", state["messages"][-1].content)
        ]
    })

    # 获取回答
    answer = response["messages"][-1].content
    writer({"travel_result": answer})

    return {"messages": [HumanMessage(content=answer)], "type": "done"}

# ------------------------------
# 交通知识问答（RAG）
# ------------------------------
def couplet_node(state: State):
    print(">>> couplet_node")
    writer = get_stream_writer()
    writer({"node": ">>>> couplet_node"})

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
你是一个专业的问答助手，必须根据提供的PDF文档内容回答问题，不要编造信息。
文档片段：{samples}
"""),
        ("user", "{text}")
    ])

    query = state["messages"][-1].content

    try:
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function=len)
        chunks = splitter.split_documents(docs)

        embeddings = DashScopeEmbeddings(
            model=EMBEDDING_MODEL,
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        samples = [doc.page_content for doc in retriever.invoke(query)]
    except Exception as e:
        samples = ["未找到相关文档内容"]
        print(f"⚠️ RAG加载失败：{e}")

    prompt = prompt_template.invoke({"samples": samples, "text": query})
    response = llm.invoke(prompt)
    writer({"couplet_result": response.content})
    return {"messages": [HumanMessage(content=response.content)], "type": "done"}
