from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

HISTORY_TURNS = 3

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

store = Chroma(
    collection_name="rivergate_support",
    persist_directory="chroma_db",
    embedding_function=embeddings,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_tokens=512,
    thinking_level="minimal",
)

history = []


def format_history(turns):
    return "\n".join(f"Employee: {question}\nAssistant: {answer}" for question, answer in turns)


def condense(question, turns):
    if not turns:
        return question

    prompt = f"""Rewrite the follow-up question as a standalone question that makes sense on its own.
Resolve any pronouns or references using the conversation. Return only the rewritten question.

Conversation so far:
{format_history(turns)}

Follow-up question: {question}

Standalone question:"""

    return llm.invoke(prompt).text.strip()


def answer(question, turns):
    search_query = condense(question, turns)
    documents = store.similarity_search(search_query, k=3)
    evidence = "\n\n".join(document.page_content for document in documents)

    prompt = f"""You are the IT helpdesk assistant for Rivergate Software.
Answer the question using only the support notes below.
If the notes do not cover it, say you don't know.

Support notes:
{evidence}

Question: {search_query}"""

    reply = llm.invoke(prompt).text.strip()
    return search_query, documents, reply


print("Rivergate IT helpdesk. Ask a question, or type quit to exit.")

while True:
    try:
        question = input("\nEmployee: ").strip()
    except (EOFError, KeyboardInterrupt):
        break

    if question.lower() in {"quit", "exit"}:
        break

    if not question:
        continue

    search_query, documents, reply = answer(question, history)

    if search_query != question:
        print(f"  [searched for] {search_query}")
    print(f"  [retrieved] {', '.join(document.metadata['source'] for document in documents)}")
    print(f"Assistant: {reply}")

    history.append((question, reply))
    history = history[-HISTORY_TURNS:]