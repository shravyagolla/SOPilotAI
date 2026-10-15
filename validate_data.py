
import json
from pathlib import Path

file_path = Path("data/cleaned_safety_procedures.json")

with file_path.open("r", encoding="utf-8") as file:
    records = json.load(file)

required_fields = [
    "procedureId",
    "title",
    "description",
    "category",
    "steps",
    "lastUpdated"
]

valid_records = 0

for record in records:
    if all(field in record for field in required_fields):
        if record["title"] and record["steps"]:
            valid_records += 1

print("Total cleaned records:", len(records))
print("Records with required fields:", valid_records)
print("Dataset validation:", "PASSED" if valid_records == len(records) else "FAILED")

if records:
    print("\nExample procedure:", records[0]["title"])
    print("Category:", records[0]["category"])
    print("Number of steps:", len(records[0]["steps"]))
