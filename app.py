
import csv
from datetime import datetime
from pathlib import Path

import streamlit as st
from sopilot_assistant import ask_sopilot

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

APP_LOG = DATA_DIR / "app_logs.csv"
FEEDBACK_LOG = DATA_DIR / "feedback_log.csv"


def append_csv(file_path, fieldnames, row):
    file_exists = file_path.exists()

    with file_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists or file_path.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(row)


st.set_page_config(
    page_title="SOPilot AI",
    page_icon="🏭",
    layout="centered",
)

st.title("🏭 SOPilot AI")
st.subheader("Manufacturing SOP Assistant")

st.write(
    "Search sample manufacturing procedures and public safety references."
)

st.warning(
    "Prototype only. Sample records are unverified, and public references "
    "are not employer-approved SOPs. Follow your organization's approved "
    "procedures for safety-critical work."
)

question = st.text_input(
    "Enter your question",
    placeholder="What hazards should be checked before entering a confined space?",
)

if st.button("Ask SOPilot AI", type="primary"):
    if not question.strip():
        st.info("Please enter a question.")
    else:
        with st.spinner("Searching available documents..."):
            try:
                answer = ask_sopilot(question.strip())
                timestamp = datetime.now().astimezone().isoformat(
                    timespec="seconds"
                )

                append_csv(
                    APP_LOG,
                    [
                        "timestamp",
                        "question",
                        "sample_procedures_returned",
                        "osha_reference_returned",
                        "answer_status",
                    ],
                    {
                        "timestamp": timestamp,
                        "question": question.strip(),
                        "sample_procedures_returned": (
                            "yes"
                            if "Procedure ID:" in answer
                            else "no"
                        ),
                        "osha_reference_returned": (
                            "yes"
                            if "Source:" in answer
                            and "osha_confined_space_quick_card.pdf" in answer
                            else "no"
                        ),
                        "answer_status": (
                            "no_match"
                            if answer.startswith(
                                "I couldn't find a matching procedure"
                            )
                            else "returned"
                        ),
                    },
                )

                st.session_state["sopilot_answer"] = answer
                st.session_state["sopilot_question"] = question.strip()
                st.session_state["sopilot_timestamp"] = timestamp
                st.session_state.pop("sopilot_feedback_saved", None)

            except Exception as error:
                st.error(f"Something went wrong: {error}")

answer = st.session_state.get("sopilot_answer")

if answer:
    st.markdown("### AI-generated answer")

    ai_answer = answer.split("Procedure: ", 1)[0]
    ai_answer = ai_answer.replace("AI answer:\n", "", 1)
    ai_answer = ai_answer.split("Retrieved sources:", 1)[0].strip()

    st.write(ai_answer)

    st.markdown("### Retrieved information")

    sample_procedures = []
    reference_documents = []
    safety_note = ""

    current_procedure = None
    current_reference = None
    reading_steps = False
    reading_excerpt = False

    for line in answer.splitlines():
        if line in (
            "Procedure: Sample dataset matches",
            "Procedure: Public reference documents (OSHA)",
        ):
            continue

        if line.startswith("Safety note:"):
            if current_procedure:
                sample_procedures.append(current_procedure)
                current_procedure = None
            if current_reference:
                reference_documents.append(current_reference)
                current_reference = None
            safety_note = line.replace("Safety note:", "", 1).strip()
            reading_steps = False
            reading_excerpt = False
            continue

        if line.startswith("Procedure: "):
            if current_procedure:
                sample_procedures.append(current_procedure)
            current_procedure = {
                "title": line.replace("Procedure: ", "", 1).strip(),
                "id": "",
                "category": "",
                "steps": [],
            }
            reading_steps = False
            reading_excerpt = False
            continue

        if line.startswith("Source: "):
            if current_procedure:
                sample_procedures.append(current_procedure)
                current_procedure = None
            if current_reference:
                reference_documents.append(current_reference)
            current_reference = {
                "source": line.replace("Source: ", "", 1).strip(),
                "page": "",
                "excerpt": [],
            }
            reading_steps = False
            reading_excerpt = False
            continue

        if current_procedure is not None:
            if line.startswith("Procedure ID:"):
                current_procedure["id"] = line.split(":", 1)[1].strip()
            elif line.startswith("Category:"):
                current_procedure["category"] = line.split(":", 1)[1].strip()
            elif line.strip() == "Sample steps:":
                reading_steps = True
                reading_excerpt = False
            elif line.strip() == "Reference excerpt:":
                reading_steps = False
                reading_excerpt = False
            elif reading_steps and line.strip():
                current_procedure["steps"].append(line.strip())

        elif current_reference is not None:
            if line.startswith("Page:"):
                current_reference["page"] = line.split(":", 1)[1].strip()
            elif line.strip() == "Reference excerpt:":
                reading_excerpt = True
            elif reading_excerpt and line.strip():
                current_reference["excerpt"].append(line.strip())

    if current_procedure:
        sample_procedures.append(current_procedure)
    if current_reference:
        reference_documents.append(current_reference)

    if sample_procedures:
        st.markdown("#### Sample dataset matches")

        for procedure in sample_procedures:
            with st.container(border=True):
                st.markdown(f"**{procedure['title']}**")

                if procedure["id"]:
                    st.caption(f"Procedure ID: {procedure['id']}")

                if procedure["category"]:
                    st.caption(f"Category: {procedure['category']}")

                if procedure["steps"]:
                    st.markdown("**Sample steps**")
                    formatted_steps = []

                    for step in procedure["steps"]:
                        if ". " in step and step.split(". ", 1)[0].isdigit():
                            formatted_steps.append(step)
                        else:
                            formatted_steps.append(
                                f"{len(formatted_steps) + 1}. {step}"
                            )

                    st.markdown("\n".join(formatted_steps))
                else:
                    st.info("No sample steps were returned for this procedure.")

    if reference_documents:
        st.markdown("#### Public reference documents (OSHA)")

        for reference in reference_documents:
            with st.container(border=True):
                st.markdown(f"**Source:** {reference['source']}")

                if reference["page"]:
                    st.caption(f"Page: {reference['page']}")

                if reference["excerpt"]:
                    st.markdown("**Reference excerpt**")
                    st.write(" ".join(reference["excerpt"]))

   
    if not sample_procedures and not reference_documents:
        st.caption("No supporting procedure or reference passages were retrieved.")


    if safety_note:
        st.warning(safety_note)

    st.caption(
        "Verify all information against approved workplace procedures "
        "before using it for safety-critical work."
    )

    st.divider()
    st.markdown("### Was this answer helpful?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "👍 Helpful",
            use_container_width=True,
            key="helpful_button",
        ):
            if not st.session_state.get("sopilot_feedback_saved"):
                append_csv(
                    FEEDBACK_LOG,
                    ["timestamp", "question", "feedback"],
                    {
                        "timestamp": datetime.now().astimezone().isoformat(
                            timespec="seconds"
                        ),
                        "question": st.session_state["sopilot_question"],
                        "feedback": "helpful",
                    },
                )
                st.session_state["sopilot_feedback_saved"] = True
                st.session_state["sopilot_feedback_value"] = "helpful"

    with col2:
        if st.button(
            "👎 Not helpful",
            use_container_width=True,
            key="not_helpful_button",
        ):
            if not st.session_state.get("sopilot_feedback_saved"):
                append_csv(
                    FEEDBACK_LOG,
                    ["timestamp", "question", "feedback"],
                    {
                        "timestamp": datetime.now().astimezone().isoformat(
                            timespec="seconds"
                        ),
                        "question": st.session_state["sopilot_question"],
                        "feedback": "not helpful",
                    },
                )
                st.session_state["sopilot_feedback_saved"] = True
                st.session_state["sopilot_feedback_value"] = "not helpful"

    if st.session_state.get("sopilot_feedback_saved"):
        st.success(
            f"Thank you! Your feedback "
            f"({st.session_state['sopilot_feedback_value']}) was saved."
        )

st.divider()
st.caption("SOPilot AI · Local prototype")
