
from pathlib import Path
from collections import Counter

PROJECT_DIR = Path(__file__).resolve().parent
LOG_FILE = PROJECT_DIR / "logs" / "sopilot.log"


def main():
    if not LOG_FILE.exists():
        print("Log file not found. Run SOPilot AI first.")
        return

    lines = LOG_FILE.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    counts = Counter()

    for line in lines:
        if "| INFO |" in line and "Received a question" in line:
            counts["questions_received"] += 1

        if "| INFO |" in line and "Answer generated" in line:
            counts["answers_generated"] += 1

        if "| ERROR |" in line:
            counts["errors"] += 1

    print("=" * 45)
    print("SOPilot AI - Monitoring Summary")
    print("=" * 45)
    print(f"Log file: {LOG_FILE}")
    print(f"Total log entries: {len(lines)}")
    print(f"Questions received: {counts['questions_received']}")
    print(f"Answers generated: {counts['answers_generated']}")
    print(f"Error entries: {counts['errors']}")


if __name__ == "__main__":
    main()
