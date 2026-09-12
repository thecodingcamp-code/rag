from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BATCH_SIZE = 8

splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=80)

chunks = []
ids = []

for path in sorted(Path("knowledge_base").glob("*.txt")):
    note = path.read_text(encoding="utf-8")
    title, _, body = note.partition("\n")
    for position, piece in enumerate(splitter.split_text(body.strip())):
        chunks.append(
            Document(
                page_content=f"{title}\n\n{piece}",
                metadata={"source": path.name, "title": title, "position": position},
            )
        )
        ids.append(f"{path.stem}-{position}")

print(f"Split 6 notes into {len(chunks)} chunks.")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

store = Chroma(
    collection_name="rivergate_support",
    persist_directory="chroma_db",
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"},
)

for start in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[start:start + BATCH_SIZE]
    batch_ids = ids[start:start + BATCH_SIZE]
    store.add_documents(documents=batch, ids=batch_ids)
    print(f"Indexed {start + len(batch)} of {len(chunks)} chunks.")

stale = set(store.get()["ids"]) - set(ids)

if stale:
    store.delete(ids=list(stale))
    print(f"Removed {len(stale)} stale chunks.")

print(f"Collection now holds {store._collection.count()} chunks.")
