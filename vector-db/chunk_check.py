from pathlib import Path

notes = [path.read_text(encoding="utf-8") for path in sorted(Path("knowledge_base").glob("*.txt"))]


def fixed_size_chunks(text, size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def title_aware_chunks(text):
    title, _, body = text.partition("\n\n")
    return [f"{title}\n\n{paragraph}" for paragraph in body.split("\n\n") if paragraph.strip()]


def chunk_everything(strategy):
    chunks = []
    for note in notes:
        chunks.extend(strategy(note))
    return chunks


questions = [
    {
        "question": "Will IT replace my battery if it barely holds a charge?",
        "needs": ["Battery health is the exception", "swap the battery free of charge"],
    },
    {
        "question": "Can a contractor get VPN access?",
        "needs": ["Connecting to the Rivergate VPN", "manager has to request it"],
    },
    {
        "question": "How long does a battery swap take?",
        "needs": ["Battery swaps are done in the office", "one working day"],
    },
]


def answerable(chunks, needs):
    return any(all(phrase in chunk for phrase in needs) for chunk in chunks)


strategies = {
    "fixed 250, no overlap": chunk_everything(lambda note: fixed_size_chunks(note, 250, 0)),
    "fixed 250, overlap 60": chunk_everything(lambda note: fixed_size_chunks(note, 250, 60)),
    "title-aware paragraphs": chunk_everything(title_aware_chunks),
}

for name, chunks in strategies.items():
    sizes = [len(chunk) for chunk in chunks]
    print(f"\n{name}: {len(chunks)} chunks, {min(sizes)}-{max(sizes)} chars")
    for item in questions:
        status = "FOUND  " if answerable(chunks, item["needs"]) else "MISSING"
        print(f"  {status} {item['question']}")
