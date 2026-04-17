# RAG_QA_System（基于 Redis 向量库 + Ollama 本地大模型）

这是一个中文 RAG 问答项目：

- **索引**：读取 `./档案库/` 下的 PDF，切分成 chunk 后写入 **Redis Stack Vector**（向量库）。
- **检索**：融合 **BM25** 与 **向量检索**，得到候选证据。
- **重排（可选）**：支持使用 DashScope 的 `qwen3-rerank` 做重排；未配置 `dashscope.api_key` 时会自动回退为“原始顺序”。
- **回答**：根据“是否包含个人敏感信息”做路由：
  - **不含敏感信息**：走在线大模型（`llm_online`）
  - **含敏感信息**：走本地 Ollama 大模型（`llm_offline`）
- **多轮对话记忆**：追加保存到本地 `memory.jsonl`，并在提示词中仅用于理解上下文（不作为证据）。

> 说明：项目中的本地大模型通过 **Ollama** 部署；Redis Stack 通过 **Docker** 拉取运行。

## 目录结构

```
RAG_QA_System/
  ├─ 档案库/                  # 你的 PDF 文档目录（自行创建/放入）
  ├─ RAG_indexing.py          # 构建索引：PDF → 切分 → Redis 向量库
  ├─ RAG_retrieving.py        # 交互式问答入口：融合检索 →（可选重排）→ 回答
  ├─ rerank_processing.py     # 重排与证据打包
  ├─ context_processing.py    # 本地对话记忆（JSONL）
  ├─ all_models.py            # Embedding / 在线 LLM / 本地 LLM
  ├─ config.json              # 记忆文件与历史轮数配置
  ├─ requirements.txt
  └─ README.md
```

## 环境要求

- Windows 10/11（你当前环境为 Windows）
- Python 3.10+（建议 3.10/3.11）
- Docker Desktop（用于 Redis Stack）
- Ollama（用于本地 Embedding 与本地 LLM）

## 1. 启动 Redis Stack（Docker）

你的代码使用 Redis 地址 `redis://localhost:6380`，因此这里把容器的 6379 映射到本机 6380。

```bash
docker run -d --name redis-stack ^
  -p 6380:6379 ^
  -p 8001:8001 ^
  redis/redis-stack:latest
```

- Redis 服务：`localhost:6380`
- RedisInsight（可视化）：`http://localhost:8001`

## 2. 启动 Ollama，并准备模型

安装并启动 Ollama 后，项目默认用到：

- Embedding：`nomic-embed-text`（用于向量化）
- 路由模型（意图识别）：`deepseek-r1:1.5b`
- 本地回答模型：`llama3.1:latest`

在终端执行：

```bash
ollama pull nomic-embed-text
ollama pull deepseek-r1:1.5b
ollama pull llama3.1
```

Ollama 默认地址为 `http://localhost:11434`（项目里也按这个写死）。

## 3. 安装 Python 依赖

建议使用虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 4. 配置（在线模型 / 重排可选）

### 4.1 对话记忆配置

`config.json` 已提供默认值：

```json
{
  "MEMORY_FILE": "memory.jsonl",
  "MAX_HISTORY_TURNS": 6
}
```

### 4.2 在线大模型（必填，否则“不含敏感信息”分支会失败）

在 `all_models.py` 中配置 `ark_api_key`（火山方舟 OpenAI 兼容接口）：

- 文件：`all_models.py`
- 变量：`ark_api_key = SecretStr("在这里填你的 key")`

并确认：

- `base_url`：`https://ark.cn-beijing.volces.com/api/v3`
- `model`：`deepseek-v3-2-251201`

如果你不打算用在线模型，最简单的做法是把 `llm_online` 也改成本地 `ChatOllama(...)`。

### 4.3 重排（可选）

如果你要启用 `qwen3-rerank`，请在代码运行前设置：

```python
import dashscope
dashscope.api_key = "你的 DashScope Key"
```

不配置也能运行：项目会自动回退为不重排（取前 10 条候选证据）。

## 5. 构建索引（把 PDF 写入向量库）

把你的 PDF 放到 `./档案库/`（与脚本同级）。

运行：

```bash
python RAG_indexing.py
```

运行后会生成：

- `all_docs.pkl`：切分后的文档块（供 BM25 检索使用）
- Redis 向量索引：`cru-index-nomic`

## 6. 启动问答（多轮对话）

运行：

```bash
python RAG_retrieving.py
```

进入交互模式后输入问题即可，输入 `bye` 退出。

## 常见问题

### 1）报错找不到 `./档案库/`

请手动创建并放入 PDF：

```
RAG_QA_System/档案库/*.pdf
```

### 2）Redis 连接失败

确认 Docker 容器已运行，并且端口映射为 `6380:6379`，与代码一致：

- 代码：`redis://localhost:6380`
- Docker：`-p 6380:6379`

### 3）Ollama 连接失败 / 模型不存在

确认 Ollama 服务在 `11434` 端口，并已 `ollama pull` 对应模型。

### 4）在线模型分支报鉴权错误

说明 `all_models.py` 里 `ark_api_key` 为空或不正确；请填入你的 key，或把 `llm_online` 改成本地模型。

## 上传到 GitHub（推荐流程）

在项目目录下执行：

```bash
git init
git add .
git commit -m "init: rag qa system"
git branch -M main
git remote add origin <你的仓库地址>
git push -u origin main
```

`.gitignore` 已忽略：

- `.idea/`
- `all_docs.pkl`
- `memory.jsonl`
- `.env*`

