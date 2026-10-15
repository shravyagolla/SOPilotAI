
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "cleaned_safety_procedures.json"
CHUNKS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "osha_confined_space_chunks.json"
)


# --------------------------------------------------
# 1. Load existing sample procedures
# --------------------------------------------------

with DATA_FILE.open("r", encoding="utf-8") as file:
    procedures = json.load(file)

titles = [p["title"] for p in procedures]

documents = [
    " ".join([
        p["title"],
        p["category"],
        p["description"],
        " ".join(s["description"] for s in p["steps"])
    ])
    for p in procedures
]

title_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

body_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

title_vectors = title_vectorizer.fit_transform(titles)
body_vectors = body_vectorizer.fit_transform(documents)


# --------------------------------------------------
# 2. Load OSHA reference-document chunks
# --------------------------------------------------

if CHUNKS_FILE.exists():
    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        osha_chunks = json.load(file)
else:
    osha_chunks = []

osha_vectorizer = None
osha_vectors = None

if osha_chunks:
    osha_texts = [chunk["text"] for chunk in osha_chunks]

    osha_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    osha_vectors = osha_vectorizer.fit_transform(osha_texts)


# --------------------------------------------------
# 3. Search existing sample procedures
# --------------------------------------------------

def search_procedures(question, top_n=2):
    title_scores = cosine_similarity(
        title_vectorizer.transform([question]),
        title_vectors
    ).flatten()

    body_scores = cosine_similarity(
        body_vectorizer.transform([question]),
        body_vectors
    ).flatten()

    scores = 0.7 * title_scores + 0.3 * body_scores
    question_lower = question.lower()

    entry_question = any(
        word in question_lower
        for word in ["entering", "entry", "enter a confined space"]
    )

    if entry_question:
        for i, procedure in enumerate(procedures):
            title = procedure["title"].lower()

            if "entry procedure" in title:
                scores[i] += 0.5
            elif "rescue" in title:
                scores[i] -= 0.2

    atmosphere_question = any(
        word in question_lower
        for word in [
            "atmosphere",
            "atmospheric",
            "gas testing",
            "hazardous gas",
            "hazardous gases",
            "gas detection",
        ]
    )

    if atmosphere_question:
        for i, procedure in enumerate(procedures):
            if "atmospheric testing" in procedure["title"].lower():
                scores[i] += 2.0

    communication_question = any(
        word in question_lower
        for word in ["communication", "communicate", "radio", "contact"]
    )

    if communication_question:
        for i, procedure in enumerate(procedures):
            if "communication protocol" in procedure["title"].lower():
                scores[i] += 0.7

    ranked_indices = scores.argsort()[::-1]
    results = []

    for index in ranked_indices:
        if scores[index] > 0:
            results.append(
                (procedures[index], float(scores[index]))
            )

        if len(results) >= top_n:
            break

    return results


# --------------------------------------------------
# 4. Search public OSHA reference documents
# --------------------------------------------------

def search_osha_documents(question, top_n=3):
    if not osha_chunks or osha_vectorizer is None:
        return []

    question_vector = osha_vectorizer.transform([question])

    scores = cosine_similarity(
        question_vector,
        osha_vectors
    ).flatten()

    ranked_indices = scores.argsort()[::-1]
    results = []

    for index in ranked_indices:
        score = float(scores[index])

        # Avoid returning irrelevant zero-score chunks.
        if score <= 0:
            continue

        chunk = osha_chunks[index]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "source_file": chunk["source_file"],
            "page": chunk["page"],
            "source_type": chunk["source_type"],
            "text": chunk["text"],
            "score": score
        })

        if len(results) >= top_n:
            break

    return results


# --------------------------------------------------
# 5. Test both search systems from the terminal
# --------------------------------------------------



if __name__ == "__main__":
    question = "What hazards should be checked before entering a confined space?"

    print("\n--- OSHA retrieval test ---")

    reference_results = search_osha_documents(question, top_n=3)

    if not reference_results:
        print("No matching public-reference passages found.")
    else:
        for result in reference_results:
            print(f"\nSource: {result['source_file']}")
            print(f"Page: {result['page']}")
            print(f"Match score: {result['score']:.3f}")
            print("Excerpt:")
            print(result["text"][:1000])
