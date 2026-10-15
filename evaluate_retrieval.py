
from retrieve_procedure import search_procedures



test_cases = [
    {
        "question": "What steps should be followed before entering a confined space?",
        "expected_title": "Confined Space Entry Procedure",
    },
    {
        "question": "How should communication be maintained in a confined space?",
        "expected_title": "Confined Space Communication Protocol",
    },
    {
        "question": "How do we test the atmosphere in a confined space?",
        "expected_title": "Confined Space Atmospheric Testing",
    },
    {
        "question": "What should be done for confined space rescue?",
        "expected_title": "Confined Space Rescue Procedure",
    },
    {
        "question": "What is the procedure for advanced confined space safety?",
        "expected_title": "Advanced Confined Space Safety",
    },
    {
        "question": "What checks are needed before space entry?",
        "expected_title": "Confined Space Entry Procedure",
    },
    {
        "question": "How can workers communicate during confined space work?",
        "expected_title": "Confined Space Communication Protocol",
    },
    {
        "question": "How can hazardous gases be detected before entry?",
        "expected_title": "Confined Space Atmospheric Testing",
    },
    {
        "question": "What is the rescue procedure for a confined space emergency?",
        "expected_title": "Confined Space Rescue Procedure",
    },
    {
        "question": "What safety procedure covers advanced confined space work?",
        "expected_title": "Advanced Confined Space Safety",
    },
]



correct = 0

print("SOPilot AI — Retrieval Evaluation")
print("=" * 40)

for case in test_cases:
    results = search_procedures(case["question"], top_n=3)

    if results:
        predicted_title = results[0][0]["title"]
        passed = predicted_title == case["expected_title"]
    else:
        predicted_title = "No match"
        passed = False

    if passed:
        correct += 1

    print("\nQuestion:", case["question"])
    print("Expected:", case["expected_title"])
    print("Retrieved:", predicted_title)
    print("Result:", "PASS" if passed else "CHECK")

accuracy = correct / len(test_cases) * 100

print("\n" + "=" * 40)
print("Correct top-ranked results:", correct, "out of", len(test_cases))
print(f"Top-1 title-match rate: {accuracy:.1f}%")
print("Note: This is a small prototype test, not a safety validation.")

# Save evaluation summary for project documentation.
from pathlib import Path

report_path = Path("data/training/retrieval_evaluation.txt")
report_path.parent.mkdir(parents=True, exist_ok=True)

with report_path.open("w", encoding="utf-8") as report:
    report.write("SOPilot AI - Retrieval Evaluation\n")
    report.write(f"Test questions: {len(test_cases)}\n")
    report.write(f"Correct top-ranked results: {correct}\n")
    report.write(f"Top-1 title-match rate: {accuracy:.1f}%\n")
    report.write(
        "Limitation: Small prototype test; not safety validation.\n"
    )

print("Evaluation report saved to:", report_path)
