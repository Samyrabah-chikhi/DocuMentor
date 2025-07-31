# Loaders
from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document

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

- Divide the content into logical sections based on major events, themes, or character developments ( it could be from 3 to 20 sections).
- For each section:
    - Write a **short, descriptive title** that reflects the main idea or turning point.
    - Below the title, include **2–4 bullet points** (using `-`) that summarize the key developments.
    - The bullet points should be:
        - Chronologically ordered (when applicable)
        - Clear and concise (no copied text from summaries)
        - Focused on important actions, motivations, decisions, or themes

---

**Part 2: Final Summary**

- After the bullet points, write a **verbose final summary** of the entire story.
- This should:
    - Read like a full retelling of the book in paragraph form
    - Emphasize cause and effect, character motivation, and narrative arc
    - Help someone understand the full story without reading the book

Here are the summaries:
```{text}```

STRUCTURED OUTLINE:
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


def load_documents(path=CONTEXT_PATH):
    loader = DirectoryLoader(path, glob="**/*.pdf", show_progress=True)
    return loader.load()


def get_text():
    text = ""
    pages = load_documents()
    for page in pages:
        text = text + page.page_content
    return text.replace("\t", " ")


def create_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=3000
    )
    return text_splitter.create_documents([docs])


def cluster_init(vectors):
    return KMeans(n_clusters=CLUSTER_NUM, random_state=42).fit(vectors)


# Function to count tokens of a split by its index
def count_tokens_for_index(splits, index):
    content = splits[index].page_content
    return len(tokenizer.encode(content, truncation=False))


if __name__ == "__main__":

    print("Loading Embedding Model...")
    embeddings = load_embeddings()

    print("Building Vector Store...")
    vector_store = build_vector_store(embeddings)

    print("Loading Documents Into A Text...")
    text = get_text()

    print("Creating Document Splits...")
    splits = create_documents(text)

    print(f"len(splits): {len(splits)}")

    print("Indexing Documents")
    vector_store.add_documents(splits)

    print("Initializing Cluster Algorithm...")
    vectors = [embeddings.embed_query(doc.page_content) for doc in splits]
    kmeans = cluster_init(np.array(vectors))

    closest_indices = []

    for i in range(CLUSTER_NUM):
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
        with open(f"summary{i}.txt", "w") as file:
            file.write(str(chunk_summary))

    summaries = "\n".join(summary_list)
    summaries = Document(page_content=summaries)

    reduce_chain = create_reduce_chain()
    output = reduce_chain.run([summaries])
    print("Output: ")
    print(output)
    with open("Summary_main.txt", "w") as file:
        file.write(str(output))
        

