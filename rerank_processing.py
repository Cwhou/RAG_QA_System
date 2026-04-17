import dashscope
from pathlib import Path
import re
import jieba

def text_rerank(inputs: dict): # 重排功能
    question = inputs["question"]
    documents = inputs["docs"]
    doc_texts = [document.page_content for document in documents]

    api_key = (getattr(dashscope, "api_key", None) or "").strip()
    # 允许不配置重排：没 key / 调用失败就直接回退到原始顺序
    if not api_key:
        return documents[:10]

    resp = dashscope.TextReRank.call(
        model="qwen3-rerank",
        api_key=api_key,
        query=question,
        documents=doc_texts,
        top_n=10,
        return_documents=True,
        instruct="Given a web search query, retrieve relevant passages that answer the query."
    )

    results = resp.output.results
    reranked_docs = [documents[item.index] for item in results] # 按index映射回原始Document

    return reranked_docs

def retrieve_and_rerank(query: str, ensemble_retriever):
    combined_docs = ensemble_retriever.invoke(query)  # 融合检索

    try:
        reranked_docs = text_rerank({  # 重排
            "question": query,
            "docs": combined_docs
        })
    except Exception:
        reranked_docs = combined_docs[:10]

    return reranked_docs

STOPWORDS = {
    "的", "了", "和", "与", "及", "或", "在", "对", "等", "中", "为", "按",
    "应", "应当", "进行", "有关", "相关", "包括", "其中", "可以", "需要", "必须",
    "一个", "一些", "以及", "通过", "根据", "对于", "不少于"
}
def remove_stopwords(text: str) -> str: # 使用jieba分词删除停用词
    # 去除多余空白
    text = re.sub(r"\s+", " ", text).strip()
    # 中文分词
    words = jieba.lcut(text)
    # 删除停用词 但保留数字、课程名等有用信息
    filtered_words = [w for w in words if w.strip() and w not in STOPWORDS]
    return "".join(filtered_words)
def pack_docs_for_compression(docs): # 格式化
    # print(docs)
    formatted = []

    # enumerate(docs, 1) 遍历docs这个列表，同时给每个元素自动编号，编号从1开始
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        source_name = Path(source).name
        page_label = doc.metadata.get("page_label", "未知页")

        compressed_content = remove_stopwords(doc.page_content)

        formatted.append(
            f"""[证据{i}]
            来源文件: {source_name}
            页码: {page_label}
            内容: {compressed_content}"""
        )

    return "\n\n".join(formatted)