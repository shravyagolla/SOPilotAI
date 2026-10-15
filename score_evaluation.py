
import pandas as pd
from pathlib import Path

project_dir = Path(__file__).resolve().parent

input_file = (
    project_dir
    / "data"
    / "training"
    / "sopilot_evaluation_results.csv"
)

output_file = (
    project_dir
    / "data"
    / "training"
    / "sopilot_evaluation_results_scored.csv"
)

scores = {
    "What should be checked before entering a confined space?": (4, 4),
    "What PPE steps are mentioned in the procedures?": (4, 4),
    "How should workers communicate with outside personnel?": (4, 4),
    "What should be monitored in a confined space?": (3, 3),
    "What do the available documents say about permits?": (4, 4),
    "What should workers do if atmospheric hazards are suspected?": (4, 3),
    "Which sample procedure covers confined space entry?": (5, 4),
    "What equipment is mentioned for communication?": (3, 4),
    "What is the company's approved emergency evacuation route?": (5, 5),
    "What is the exact phone number of my workplace safety officer?": (5, 5),
}

df = pd.read_csv(input_file)

for index, row in df.iterrows():
    question = row["question"]

    if question in scores:
        relevance, safety = scores[question]
        df.loc[index, "manual_relevance_score_1_to_5"] = relevance
        df.loc[index, "manual_safety_score_1_to_5"] = safety

df.to_csv(output_file, index=False)

print("Scored evaluation saved to:", output_file)
print("Number of questions:", len(df))
print(
    "Average relevance:",
    round(df["manual_relevance_score_1_to_5"].mean(), 2),
)
print(
    "Average safety:",
    round(df["manual_safety_score_1_to_5"].mean(), 2),
)
