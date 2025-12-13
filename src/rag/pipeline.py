import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CHROMA_DB_DIR, EMBEDDING_MODEL, LLM_MODEL, TEMPERATURE, TOP_K

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Retriever
embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL)

vectorstore = Chroma(
    collection_name="taiwan_attractions",
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": TOP_K}
)

# LLM
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL, api_key=os.getenv("GOOGLE_API_KEY"), temperature=TEMPERATURE
)

# Prompt Template
template = """你是一個專業的台灣旅遊顧問助手。請根據以下提供的上下文資訊來回答使用者的問題。
上下文資訊:
{context}

使用者問題: {question}

請提供詳細且實用的回答。如果上下文中沒有相關資訊,請誠實地說你不知道,不要編造答案。
回答:"""

prompt = PromptTemplate.from_template(template=template)

# RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def ask(question: str):
    return rag_chain.invoke(question)
