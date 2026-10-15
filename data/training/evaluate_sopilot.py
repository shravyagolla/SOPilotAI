
import sys
from pathlib import Path

# Locate the main SOPilotAI project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from retrieve_procedure import search_procedures, search_osha_documents


TEST_CASES = [
    {
        "question": "What hazards should be checked before entering a confined space?",
        "expected_procedure": "CONF-013",
        "expect_osha": True,
    },
    {
        "question": "How should the atmosphere be monitored?",
        "expected_procedure": "CONF-025",
        "expect_osha": True,
    },
    {
        "question": "What PPE is needed for confined space entry?",
        "expected_procedure": "CONF-013",
        "expect_osha": True,
    },
    {
        "question": "How can workers communicate with outside personnel?",
        "expected_procedure": "CONF-028",
        "expect_osha": True,
    },
    {
        "question": "What permits are needed before entry?",
        "expected_procedure": "CONF-019",
        "expect_osha": True,
    },
    {
        "question": "How do I repair a broken conveyor belt motor?",
        "expected_procedure": None,
        "expect_osha": False,
    },
]


def main():
    total = len(TEST_CASES)
    procedure_hits = 0
    answerable_tests = 0
    osha_hits = 0
    no_match_checks = 0

    print("=" * 65)
    print("SOPilot AI - Retrieval Evaluation")
    print("=" * 65)

    for index, test in enumerate(TEST_CASES, start=1):
        question = test["question"]
        expected = test["expected_procedure"]

        procedures = search_procedures(question, top_n=3)
        references = search_osha_documents(question, top_n=2)

        returned_ids = [
            procedure.get("procedureId")
            for procedure, score in procedures
        ]

        if expected is not None:
            answerable_tests += 1
            procedure_correct = expected in returned_ids

            if procedure_correct:
                procedure_hits += 1
        else:
            procedure_correct = False

        no_match_correct = (
            expected is None and not procedures and not references
        )

        if expected is None and no_match_correct:
            no_match_checks += 1

        osha_correct = (
            bool(references)
            if test["expect_osha"]
            else not references
        )

        if osha_correct:
            osha_hits += 1

        print(f"\nTest {index}: {question}")
        print(f"Expected procedure: {expected or 'No matching procedure'}")
        print(f"Returned procedure IDs: {returned_ids}")
        print(f"OSHA passages found: {len(references)}")

        if expected is None:
            print(
                "No-match behavior:",
                "PASS" if no_match_correct else "CHECK",
            )
        else:
            print(
                "Expected procedure found:",
                "PASS" if procedure_correct else "CHECK",
            )

        print(
            "OSHA retrieval check:",
            "PASS" if osha_correct else "CHECK",
        )

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"Total test questions: {total}")
    print(
        f"Expected procedure found: "
        f"{procedure_hits}/{answerable_tests}"
    )
    print(f"OSHA retrieval checks passed: {osha_hits}/{total}")
    print(f"No-match checks passed: {no_match_checks}/1")

    print(
        "\nNote: This is a small, manually selected test set. "
        "Results are preliminary and are not a general accuracy guarantee."
    )


if __name__ == "__main__":
    main()
