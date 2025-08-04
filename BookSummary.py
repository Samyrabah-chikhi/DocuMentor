# API
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
# Loaders
from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader
import tempfile

# LLM
from langchain_ollama import OllamaLLM

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector store
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# Summarize using map_reduce
from langchain.chains.summarize import load_summarize_chain

# Data Science
import numpy as np
from sklearn.cluster import KMeans

# Load the tokenizer once
from transformers import AutoTokenizer

# CONFIGURATION
MODEL_NAME = "gemma3:1b"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CONTEXT_PATH = "./context"
CLUSTER_NUM = 20

# ============ FASTAPI INIT ============
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = None
embeddings = None

# ============ LLM INIT ============
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
LLM = OllamaLLM(
    model=MODEL_NAME,
    num_thread=4,
    temperature=0.3,  # Lower temperature = more focused and factual
    top_k=30,  # Moderate diversity, avoids nonsense
    top_p=0.9,  # Balanced sampling
    repeat_penalty=1.5,  # Reduces redundant phrases
    repeat_last_n=256,  # Looks farther back to prevent repetition
    mirostat=0,  # Disabled for deterministic behavior
    num_ctx=16384,
)


# HELPER FUNCTIONS
def build_vector_store(embeddings):
    dim = len(embeddings.embed_query("Hello World"))
    index = faiss.IndexFlatL2(dim)
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )


def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_map_chain():
    map_prompt = """
You will be given a passage from a book, enclosed in triple backticks (```).
Your task is to write a clear, cohesive summary of the passage in a single paragraph.

Your summary should:
- Accurately capture the main events and actions.
- Include the motivations, emotions, or key ideas expressed.
- Preserve important names, settings, and turning points.
- Avoid vague generalizations — instead, clarify what *actually* happens.

Write in a neutral, narrative tone. Think of it as explaining this scene to someone who hasn’t read it, but needs to fully understand it.

```{text}```
SUMMARY:
"""
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    return load_summarize_chain(llm=LLM, chain_type="stuff", prompt=map_prompt_template)


def create_reduce_chain():
    combine_prompt = """
You will be given a collection of summaries from a book, enclosed in triple backticks (```).

Your task has two parts:

---

**Part 1: Structured Outline**

- Divide the content into logical sections based on major events, themes, or character developments (3 to 20 sections).
- For each section:
    - Write a **short, descriptive title**.
    - Below the title, include **2–4 bullet points** using `-`, covering the main developments.

---

**Part 2: Final Summary**

- After the bullet points, write a **verbose final summary** (a few paragraphs).
- It should:
    - Retell the full story in paragraph form
    - Emphasize cause and effect, motivations, and themes
    - Help the reader understand the full story

Here are the summaries:
```{text}```

---
return the content in the following format without including any introductory or closing phrases

**STRUCTURED OUTLINE:**

[Start Part 1 here...]

---

**FINAL SUMMARY:**

[Start Part 2 here...]
"""

    combine_prompt_template = PromptTemplate(
        template=combine_prompt, input_variables=["text"]
    )
    return load_summarize_chain(
        llm=LLM,
        chain_type="stuff",
        prompt=combine_prompt_template,
        verbose=True,
        # verbose=True # Set this to true if you want to see the inner workings
    )


def load_documents_from_path(path=CONTEXT_PATH):
    loader = DirectoryLoader(path, glob="**/*.pdf", show_progress=True)
    return loader.load()


def get_text_from_path():
    text = ""
    pages = load_documents_from_path()
    for page in pages:
        text = text + page.page_content
    return text.replace("\t", " ")


def get_text_from_parameters(file: UploadFile):
    text = ""
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    # Load with PyMuPDFLoader
    loader = PyMuPDFLoader(tmp_path)
    pages = loader.load()

    for page in pages:
        text += page.page_content

    return text.replace("\t", " ")


def create_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=3000
    )
    return text_splitter.create_documents([docs])


def cluster_init(vectors, cluster_num=CLUSTER_NUM):
    return KMeans(n_clusters=cluster_num, random_state=42).fit(vectors)


# Function to count tokens of a split by its index
def count_tokens_for_index(splits, index):
    content = splits[index].page_content
    return len(tokenizer.encode(content, truncation=False))


# ============ STARTUP EVENT ============


@app.on_event("startup")
def startup():
    global vector_store, embeddings

    print("Loading Embedding Model...")
    embeddings = load_embeddings()

    print("Building Vector Store...")
    vector_store = build_vector_store(embeddings)


# ============ MAIN ROUTE ============


@app.post("/")
async def ask_question(file: UploadFile = File(...)):
    print("Loading Documents Into A Text...")
    text = get_text_from_parameters(file)

    print("Creating Document Splits...")
    splits = create_documents(text)

    print(f"len(splits): {len(splits)}")

    print("Indexing Documents")
    vector_store.add_documents(splits)

    print("Initializing Cluster Algorithm...")
    vectors = [embeddings.embed_query(doc.page_content) for doc in splits]

    cluster_num = CLUSTER_NUM
    if len(splits) < 3:
        cluster_num = 1
    elif len(splits) < 10:
        cluster_num = 3
    elif len(splits) < 25:
        cluster_num = 4
    elif len(splits) < 50:
        cluster_num = 6
    elif len(splits) < 100:
        cluster_num = 12
    else:
        cluster_num = min(20, len(splits) // 5)

    kmeans = cluster_init(np.array(vectors), cluster_num)

    closest_indices = []

    for i in range(cluster_num):
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        closest_index = np.argmin(distances)
        closest_indices.append(closest_index)

    selected_indices = sorted(closest_indices)

    for idx in selected_indices:
        token_count = count_tokens_for_index(splits, idx)
        preview = splits[idx].page_content[:200].replace("\n", " ")  # first 200 chars
        print(f"\nIndex {idx} | Tokens: {token_count}\nPreview: {preview}...\n")
    selected_docs = [splits[doc] for doc in selected_indices]

    map_chain = create_map_chain()
    summary_list = []

    for i, doc in enumerate(selected_docs):
        chunk_summary = map_chain.run([doc])
        summary_list.append(chunk_summary)
        print(f"Chunk summary {i}")
        print(chunk_summary)
        with open(f"summary{i}_{MODEL_NAME}.txt", "w") as file:
            file.write(str(chunk_summary))

    summaries = "\n".join(summary_list)
    summaries = Document(page_content=summaries)

    reduce_chain = create_reduce_chain()
    output = reduce_chain.run([summaries])
    print("Output: ")
    print(output)
    with open(f"Summary_main_{MODEL_NAME}.txt", "w") as file:
        file.write(str(output))

    return {"Summary": str(output)}


# ============ RUN ============

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("BookSummary:app", host="127.0.0.1", port=8000, reload=True)
