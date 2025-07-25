import faiss
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

# Loading LLM model
model_name = "llama3.2"
model = OllamaLLM(model=model_name)

# Loading embedding model
embedding_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_name)

# Create vector store
embedding_dim = len(embeddings.embed_query("Hello World"))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# Load Documents
loader = DirectoryLoader("./context/", glob="**/*.pdf", show_progress=True)
docs = loader.load()
print("Documents loaded succesfully")


# Text splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

all_splits = text_splitter.split_documents(docs)
print("Documents splitted succesfully")

# Store splits inside the vector store

document_ids = vector_store.add_documents(docs)

# Template for LLM
template = """
You are an expert assistant specialized in answering questions about self-development books.

Use the following context to answer the question at the end. If the answer isn't in the context, say you don't know — don’t try to make anything up.

Keep your response to a maximum of three concise sentences. Always end your answer with: "Thanks for asking!"

Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template=template)

chain = prompt | model

# Main loop
while True:
    print("--------------------------------------------\n\n")
    question = input("Ask your question (q to quit): \n\n")
    if question == "q":
        break

    # Get docs and use filter them
    context_docs = vector_store.similarity_search(question)
    context_text = "\n\n".join(doc.page_content for doc in context_docs)

    result = chain.invoke({"context": context_text, "question": question})
    print(result)
