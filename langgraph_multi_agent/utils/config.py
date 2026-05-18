import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

# 加载环境变量
load_dotenv()

# 配置常量
PDF_PATH = r"D:\python\project\learnllm\learn\langgraph_multi_agent\resources\共享电动自行车路段超速风险影响因素分析_张晓龙.pdf"
EMBEDDING_MODEL = "text-embedding-v4"
NODES = ["travel", "joke", "couplet", "other"]

# 初始化大模型
def get_llm():
    return ChatDeepSeek(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        model=os.getenv("MODEL_ID")
    )