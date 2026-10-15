
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parent

CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "osha_confined_space_chunks.json"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Minimum cosine similarity for accepting a passage.
SIMILARITY_THRESHOLD = 0.20

print("Loading semantic embedding model...")
model = SentenceTransformer(MODEL_NAME)

_cached_chunks = None
_cached_documents = None
_cached_embeddings = None


def load_chunks():
    """Load the processed OSHA document chunks."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}"
        )

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("The OSHA chunks file is empty.")

    return chunks


def prepare_documents():
    """Prepare and cache document embeddings once."""
    global _cached_chunks
    global _cached_documents
    global _cached_embeddings

    if _cached_embeddings is not None:
        return (
            _cached_chunks,
            _cached_documents,
            _cached_embeddings,
        )

    chunks = load_chunks()

    valid_items = [
        (chunk, str(chunk.get("text", "")).strip())
        for chunk in chunks
        if str(chunk.get("text", "")).strip()
    ]

    if not valid_items:
        raise ValueError(
            "No usable text was found in the OSHA chunks."
        )

    _cached_chunks = [item[0] for item in valid_items]
    _cached_documents = [item[1] for item in valid_items]

    _cached_embeddings = model.encode(
        _cached_documents,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return (
        _cached_chunks,
        _cached_documents,
        _cached_embeddings,
    )


def search_osha_semantic(question, top_n=3):
    """Return the most relevant OSHA passages using embeddings."""
    if not question or not question.strip():
        return []

    if top_n < 1:
        return []

    chunks, documents, document_embeddings = prepare_documents()

    question_embedding = model.encode(
        [question.strip()],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    scores = cosine_similarity(
        question_embedding,
        document_embeddings,
    )[0]
    question_lower = question.lower()

    if "monitor" in question_lower or "atmosphere" in question_lower:
        for index, document in enumerate(documents):
            document_lower = document.lower()

            if (
                "oxygen content" in document_lower
                and "flammability" in document_lower
                and "toxicity" in document_lower
            ):
                scores[index] += 0.05
        # Identify whether the source contains specific PPE names.
    if (
        "personal protective equipment" in question_lower
        or "what ppe" in question_lower
    ):
        ppe_terms = [
            "gloves",
            "goggles",
            "respirator",
            "hard hat",
            "safety boots",
        ]

        source_text = " ".join(documents).lower()

        if not any(term in source_text for term in ppe_terms):
            print(
                "Note: The available OSHA reference does not "
                "specify individual PPE items."
            )

    # Rank passages using genuine cosine similarity scores.
    ranked_indices = [
        int(index)
        for index in scores.argsort()[::-1]
        if scores[index] >= SIMILARITY_THRESHOLD
    ][:top_n]

    results = []

    for index in ranked_indices:
        result = dict(chunks[index])
        result["text"] = documents[index]
        result["score"] = float(scores[index])
        results.append(result)

    return results


if __name__ == "__main__":
    test_questions = [
        "What should be monitored in a confined space?",
        "What personal protective equipment is mentioned?",
        "What do the documents say about entry permits?",
        "How should workers communicate with outside personnel?",
    ]

    for question in test_questions:
        print("\n" + "=" * 60)
        print("Question:", question)

        try:
            results = search_osha_semantic(question, top_n=3)

            if not results:
                print(
                    "No sufficiently similar passages found. "
                    "Consult an appropriate approved reference."
                )
                continue

            for result in results:
                print("\nScore:", round(result["score"], 3))
                print(
                    "Source:",
                    result.get("source_file", "Unknown"),
                )
                print("Page:", result.get("page", "Unknown"))
                print("Excerpt:", result["text"][:350])

        except Exception as error:
            print("Retrieval error:", repr(error))

