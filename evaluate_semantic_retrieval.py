
from semantic_retrieval import search_osha_semantic


# expected_terms are simple checks for the retrieved excerpt.
# They are a preliminary proxy, not a substitute for human review.
TEST_CASES = [
    {
        "question": "What hazards should be checked before entering a confined space?",
        "expected_terms": ["hazard", "confined"],
        "in_scope": True,
    },
    {
        "question": "How can I identify a hazardous atmosphere in a confined space?",
        "expected_terms": ["oxygen", "atmosphere", "gas"],
        "in_scope": True,
    },
    {
        "question": "What is a permit-required confined space?",
        "expected_terms": ["permit", "confined"],
        "in_scope": True,
    },
    {
        "question": "How should workers communicate with an attendant?",
        "expected_terms": ["attendant", "communicat"],
        "in_scope": True,
    },
    {
        "question": "What should be checked before workers enter a tank?",
        "expected_terms": ["hazard", "atmosphere", "entry"],
        "in_scope": True,
    },
    {
        "question": "How can oxygen deficiency be detected?",
        "expected_terms": ["oxygen"],
        "in_scope": True,
    },
    {
        "question": "What should be done if a hazardous atmosphere is detected?",
        "expected_terms": ["hazard", "atmosphere", "evacuat"],
        "in_scope": True,
    },
    {
        "question": "Why is atmospheric testing necessary before entry?",
        "expected_terms": ["atmosphere", "testing", "oxygen"],
        "in_scope": True,
    },
    {
        "question": "What personal protective equipment may be needed?",
        "expected_terms": ["protective", "equipment", "PPE"],
        "in_scope": True,
    },
    {
        "question": "What is the role of the outside attendant?",
        "expected_terms": ["attendant"],
        "in_scope": True,
    },
    {
        "question": "How should emergency rescue be handled?",
        "expected_terms": ["rescue", "emergency"],
        "in_scope": True,
    },
    {
        "question": "What should workers do before starting confined-space work?",
        "expected_terms": ["entry", "hazard", "permit"],
        "in_scope": True,
    },
    {
        "question": "How do flammable gases create risks in enclosed spaces?",
        "expected_terms": ["flammability", "flammable", "explosive"],
        "in_scope": True,
    },
    {
        "question": "How can toxic substances affect confined-space entry?",
        "expected_terms": ["toxic", "toxicity", "hazard"],
        "in_scope": True,
    },
    {
        "question": "What information should an entry permit contain?",
        "expected_terms": ["permit"],
        "in_scope": True,
    },
    {
        "question": "How often should the atmosphere be monitored?",
        "expected_terms": ["monitor", "atmosphere", "testing"],
        "in_scope": True,
    },
    {
        "question": "How do I repair a broken conveyor belt motor?",
        "expected_terms": [],
        "in_scope": False,
    },
    {
        "question": "What is the best recipe for chocolate cake?",
        "expected_terms": [],
        "in_scope": False,
    },
    {
        "question": "How do I configure a Python web server?",
        "expected_terms": [],
        "in_scope": False,
    },
    {
        "question": "Who won the latest football match?",
        "expected_terms": [],
        "in_scope": False,
    },
]


def main():
    print("=" * 70)
    print("SOPilot AI - Semantic Retrieval Evaluation")
    print("=" * 70)

    total = len(TEST_CASES)
    relevant_total = 0
    relevant_hits = 0
    out_of_scope_total = 0
    out_of_scope_returns = 0

    for number, case in enumerate(TEST_CASES, start=1):
        question = case["question"]
        expected_terms = case["expected_terms"]
        in_scope = case["in_scope"]

        results = search_osha_semantic(question, top_n=3)

        print(f"\nTest {number}/{total}: {question}")

        if not results:
            print("Result: No matching passage")
            if in_scope:
                relevant_total += 1
            else:
                out_of_scope_total += 1
            continue

        best = results[0]
        excerpt = str(best.get("text", ""))
        searchable_text = (
            excerpt + " " + str(best.get("source_file", ""))
        ).lower()

        print("Source:", best.get("source_file", "Unknown"))
        print("Page:", best.get("page", "Unknown"))
        print("Similarity:", round(best["score"], 3))
        print("Excerpt:", excerpt[:250])

        if in_scope:
            relevant_total += 1
            matched_terms = [
                term for term in expected_terms
                if term.lower() in searchable_text
            ]

            hit = bool(matched_terms)
            relevant_hits += int(hit)

            print("Expected-term check:", "PASS" if hit else "REVIEW")
            print("Matched terms:", matched_terms)
        else:
            out_of_scope_total += 1
            out_of_scope_returns += 1
            print("Out-of-scope check: REVIEW - retrieval returned a result")

    print("\n" + "=" * 70)
    print("PRELIMINARY SUMMARY")
    print("=" * 70)
    print("Total questions:", total)
    print("In-scope questions:", relevant_total)
    print("In-scope questions with a term match:", relevant_hits)

    if relevant_total:
        print(
            "Term-match rate:",
            f"{relevant_hits / relevant_total:.1%}"
        )

    print("Out-of-scope questions:", out_of_scope_total)
    print("Out-of-scope questions returning results:", out_of_scope_returns)
    print(
        "\nImportant: These keyword checks are not formal retrieval "
        "accuracy metrics. Review the retrieved passages manually and "
        "label relevant chunks before calculating Recall@K or MRR."
    )


if __name__ == "__main__":
    main()
