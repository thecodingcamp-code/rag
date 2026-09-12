from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_tokens=512,
    thinking_level="minimal",
)

documents = []
sources = []

for path in sorted(Path("knowledge_base").glob("*.txt")):
    documents.append(path.read_text(encoding="utf-8"))
    sources.append(path.name)

print(f"Loaded {len(documents)} documents.")

document_vectors = np.array(embeddings.embed_documents(documents))
print("Embedding matrix shape:", document_vectors.shape)

question = "How many holiday days do I get in my first year?"

query_vector = np.array(embeddings.embed_query(question))

scores = document_vectors @ query_vector / (
    np.linalg.norm(document_vectors, axis=1) * np.linalg.norm(query_vector)
)

top_indices = np.argsort(scores)[::-1][:2]

print("\nRetrieved:")
for index in top_indices:
    print(f"  {sources[index]}  score {scores[index]:.3f}")

context = "\n\n".join(documents[index] for index in top_indices)

prompt = f"""You are the IT helpdesk assistant for Rivergate Software.
Answer the employee's question using only the support notes below.
If the notes do not cover it, say that you don't know.

Support notes:
{context}

Question: {question}
"""

response = llm.invoke(prompt)

print("\nAnswer:")
print(response.content)
