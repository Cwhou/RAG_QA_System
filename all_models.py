from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import SecretStr

# 文本向量化模型
embedding_model = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

# 通过api key访问的大模型客户端
ark_api_key = SecretStr("") # 改为自己的密钥
llm_online = ChatOpenAI(
    model = "deepseek-v3-2-251201",
    base_url = "https://ark.cn-beijing.volces.com/api/v3",
    api_key = ark_api_key
)

# 本地大模型
llm_offline = ChatOllama(model="llama3.1:latest", base_url="http://localhost:11434")