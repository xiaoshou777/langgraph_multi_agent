import random
import gradio as gr
from langchain_core.messages import HumanMessage
from agent.graph import graph



# 异步处理函数
async def process_input(text):
    config = {
        "configurable": {
            "thread_id": random.randint(1,1000)
        }
    }
    # 关键：异步调用 ainvoke，不是 invoke
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=text)]},
        config
    )
    return result["messages"][-1].content

with gr.Blocks() as demo:
    gr.Markdown("# LangGraph Multi-Agent")
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 可以问路线规划，交通知识科普，讲笑话，快来试试吧。")
            inputs_text = gr.Textbox(label="问题*", placeholder="请输入你的问题", value="规划从天安门到北京站的路线")
            btn_start = gr.Button("Start", variant="primary")
        with gr.Column():
            output_text = gr.Textbox(label="Output")

    # 绑定异步函数
    btn_start.click(process_input, inputs=[inputs_text], outputs=[output_text])

demo.launch()