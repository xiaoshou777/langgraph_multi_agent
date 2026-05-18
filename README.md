# LangGraph 多智能体问答系统
## 项目简介
本项目基于 **LangGraph + LangChain + MCP + RAG** 构建**分类式多智能体系统**，由总控调度节点对用户问题进行意图识别，自动分发至对应子智能体处理，集成高德地图MCP工具调用、PDF文档检索问答、趣味内容生成，前端使用Gradio实现可视化交互。

### 核心功能
1. **旅游路线规划**：调用高德地图MCP服务，实现自驾路线规划
2. **趣味笑话生成**：生成简短幽默的笑话内容
3. **交通领域文档问答**：基于PDF文档实现RAG检索增强问答，回答相关问题
4. **通用问题兜底回复**：非匹配问题统一友好兜底应答

## 项目目录结构
```
langgraph_multi_agent/
├── .env                     # 环境变量配置文件（API密钥、模型地址等）
├── main.py                  # 项目入口，Gradio Web前端服务
├── requirements.txt         # 项目依赖清单
├── agent/                   # 智能体核心模块
│   ├── __init__.py
│   ├── state.py            # 定义LangGraph全局状态
│   ├── nodes.py            # 所有业务节点实现（调度、笑话、旅游、RAG问答）
│   ├── router.py           # 节点路由逻辑
│   └── graph.py            # LangGraph工作流构建与编译
├── utils/                   # 通用工具与配置
│   ├── __init__.py
│   └── config.py           # 全局常量、LLM模型初始化、路径配置
└── resources/               # 资源文件目录
    └── xxx.pdf  # RAG知识库文档
```

## 环境要求
- Python ≥ 3.10
- 支持异步IO环境

## 快速开始
### 1. 配置环境变量
在项目根目录新建/编辑 `.env` 文件，填入以下密钥：
```env
# DeepSeek 大模型配置
API_KEY=你的DeepSeek_API_KEY
BASE_URL=https://api.deepseek.com
MODEL_ID=deepseek-chat

# 阿里云通义Embedding模型密钥
DASHSCOPE_API_KEY=你的DASHSCOPE_API_KEY
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 启动项目
```bash
python main.py
```
启动成功后，浏览器自动打开 `http://127.0.0.1:7860` 即可使用可视化对话界面。

## 核心模块说明
### 1. 状态管理（state.py）
使用TypedDict定义全局状态，包含**消息列表**、**问题分类类型**，通过消息累加器实现多轮对话上下文传递。

### 2. 调度节点（supervisor_node）
作为总控智能体，接收用户问题，通过大模型进行意图分类，输出指定标签：
- `travel`：旅游路线规划
- `joke`：讲笑话
- `couplet`：交通PDF文档问答
- `other`：其他通用问题

### 3. 业务节点
- **travel_node（异步）**：对接高德地图MCP服务，调用地图工具生成自驾路线；
- **joke_node**：调用DeepSeek生成100字以内简短笑话；
- **couplet_node**：PDF文档加载→文本分割→向量库构建→相似度检索→RAG问答；
- **other_node**：兜底回复，无法识别问题时统一应答。

### 4. 工作流逻辑
```
用户输入 → 调度节点分类 → 路由分发至对应子节点 → 节点处理 → 返回结果
```

## 技术栈
- 大模型：DeepSeek
- 框架：LangGraph、LangChain
- 向量库：FAISS
- Embedding：阿里云DashScope通义向量模型
- 工具调用：MCP（高德地图）
- 前端：Gradio
- 文档处理：PyPDF、CharacterTextSplitter

## 注意事项
1. 高德地图MCP密钥已内置，如需替换请修改 `nodes.py` 中MCP配置；
2. PDF文件路径已相对化，请勿随意修改文件名与目录；
3. 模型调用需保证网络可访问对应API地址；
4. 支持多轮对话，通过LangGraph内存检查点实现会话隔离。

