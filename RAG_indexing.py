import pickle
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import redis
from langchain_redis import RedisConfig, RedisVectorStore
from all_models import embedding_model

# 加载文档
directLoader = DirectoryLoader("./档案库/", glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
documents = directLoader.load()
print(documents)

# 切分文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100, # 保留上下文 前后均保留chunk_overlap的上下文
    separators=[
        r"\n[一二三四五六七八九十]+、",
        r"\n（[一二三四五六七八九十]+）",
        r"\n\d+、",
        r"\n\n",
        r"\n",
        r"。",
        r"；",
        r"，",
        r""
    ], # 按separators的顺序递归切分，如果切出来的块不超过chunk_size，就不再继续，若某个块仍然太大，则继续用后面的分隔符递归处理
    keep_separator=True
)

segments = text_splitter.split_documents(documents) # 全部的Document分块

# 持久化到本地
with open("all_docs.pkl", "wb") as f:
    pickle.dump(segments, f)

for segment in segments:
    print(segment.page_content)
    print("----------")

# 文本向量化
redis_url = "redis://localhost:6380"
redis_client = redis.from_url(redis_url)

config = RedisConfig(
    index_name="cru-index-nomic",
    redis_url=redis_url
)
vector_store = RedisVectorStore(embedding_model, config=config)
vector_store.add_documents(segments) # 入向量库