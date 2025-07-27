from fastapi import FastAPI
from pydantic import BaseModel
import json
import httpx
import faiss

from typing import AsyncGenerator
from fastapi.responses import StreamingResponse

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

# ============ CONFIGURATION ============
MODEL_NAME = "gemma3:1b"
OLLAMA_URL = "http://localhost:11434/api/generate"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CONTEXT_PATH = "./context/"

# ============ FASTAPI INIT ============
app = FastAPI()
vector_store = None
prompt_template = None

# ============ REQUEST MODEL ============
class QuestionRequest(BaseModel):
    question: str

# ============ HELPER FUNCTIONS ============

def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def build_vector_store(embeddings):
    dim = len(embeddings.embed_query("Hello World"))
    index = faiss.IndexFlatL2(dim)
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

def load_documents():
    loader = DirectoryLoader(CONTEXT_PATH, glob="**/*.pdf", show_progress=True)
    return loader.load()

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return splitter.split_documents(docs)

def build_prompt():
    template = """
You are an expert assistant specialized in answering questions about self-development books.

Use the following context to answer the question at the end. If the answer isn't in the context, say you don't know — don’t try to make anything up.

Keep your response to a maximum of three concise sentences. Always end your answer with: "Thanks for asking!"

Context:
{context}

Question:
{question}
"""
    return ChatPromptTemplate.from_template(template=template)

async def call_llm_stream(prompt_text: str) -> AsyncGenerator[str, None]:
    data = {"model": MODEL_NAME, "prompt": prompt_text, "stream": True}
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", OLLAMA_URL, json=data) as response:
            async for line in response.aiter_lines():
                if line:
                    chunk = json.loads(line)
                    text = chunk.get("response", "")
                    yield text  

# ============ STARTUP EVENT ============

@app.on_event("startup")
def startup():
    global vector_store, prompt_template

    print("Loading embedding model...")
    embeddings = load_embeddings()

    print("Building vector store...")
    vector_store = build_vector_store(embeddings)

    print("Loading documents...")
    docs = load_documents()

    print("Splitting documents...")
    splits = split_documents(docs)

    print("Indexing documents...")
    vector_store.add_documents(splits)

    prompt_template = build_prompt()
    print("Startup complete.")

# ============ MAIN ROUTE ============

@app.post("/")
async def ask_question(request: QuestionRequest):
    print(f"Received question: {request.question}")

    context_docs = vector_store.similarity_search(request.question)
    context_text = "\n\n".join(doc.page_content for doc in context_docs)
    print("Context retrieved")

    prompt_text = prompt_template.format(context=context_text, question=request.question)

    # Return streaming response
    return StreamingResponse(call_llm_stream(prompt_text), media_type="text/plain")

# ============ RUN ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
