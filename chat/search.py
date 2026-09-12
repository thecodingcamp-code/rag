from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

store = Chroma(
    collection_name="rivergate_support",
    persist_directory="chroma_db",
    embedding_function=embeddings,
)

question = "my laptop battery barely lasts an hour, can I get a new one?"

results = store.similarity_search_with_score(question, k=3)

print(f"Question: {question}\n")
for document, distance in results:
    print(f"{distance:.3f}  {document.metadata['source']}")
    print(f"       {document.page_content[:80]}...\n")

filtered = store.similarity_search(
    "how do I get access?",
    k=2,
    filter={"source": "vpn_access.txt"},
)

print("Filtered to vpn_access.txt:")
for document in filtered:
    print(f"  {document.page_content[:80]}...")

openers = store.similarity_search(
    "what do I need to do first?",
    k=3,
    filter={"position": 0},
)

print("\nFiltered to the opening chunk of each note:")
for document in openers:
    print(f"  {document.metadata['title']}")

stored = store.get(ids=["vpn_access-2"])

print("\nFetched by ID (vpn_access-2):")
print(f"  {stored['documents'][0][:80]}...")
