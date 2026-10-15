
from retrieve_procedure import search_procedures
from semantic_retrieval import search_osha_semantic
from load_finetuned_model import model, tokenizer
import torch

from logging_config import get_logger

logger = get_logger("sopilot_assistant")


def ask_sopilot(question):
    logger.info("Received a question")
    question = question.strip()

    if not question:
        logger.warning("Empty question received")
        return "Please enter a question."

    try:
        sample_results = search_procedures(question, top_n=2)
    except Exception:
        logger.exception("Sample procedure retrieval failed")
        sample_results = []

    try:
        reference_results = search_osha_semantic(question, top_n=2)
    except Exception:
        logger.exception("OSHA semantic retrieval failed")
        reference_results = []

    # Build context from retrieved documents.
    context_parts = []

    for procedure, score in sample_results:
        steps = procedure.get("steps", [])

        if isinstance(steps, list):
            formatted_steps = []

            for i, step in enumerate(steps):
                if isinstance(step, dict):
                    step_number = step.get("stepNumber", i + 1)
                    description = step.get("description", "")
                    formatted_steps.append(
                        f"{step_number}. {description}"
                    )
                else:
                    formatted_steps.append(f"{i + 1}. {step}")

            steps_text = "\n".join(formatted_steps)
        else:
            steps_text = str(steps)

        context_parts.append(
            f"Sample procedure: {procedure.get('title', 'Unknown')}\n"
            f"Procedure ID: {procedure.get('procedureId', 'Unknown')}\n"
            f"Category: {procedure.get('category', 'Unknown')}\n"
            f"Description: {procedure.get('description', 'Not provided')}\n"
            f"Steps:\n{steps_text}"
        )

    for reference in reference_results:
        context_parts.append(
            f"Source: {reference.get('source_file', 'Unknown')}\n"
            f"Page: {reference.get('page', 'Unknown')}\n"
            f"Excerpt: {reference.get('text', '')}"
        )

    if not context_parts:
        logger.warning("No relevant context found for the question")
        return (
            "I couldn't find relevant information in the available documents. "
            "Consult your organization's approved SOP and qualified safety "
            "personnel. Do not rely on an AI-generated answer for "
            "safety-critical work."
        )

    context = "\n\n".join(context_parts)
    question_lower = question.lower()
    context_lower = context.lower()

    # Safety check: company-specific evacuation route.
    asks_for_evacuation_route = (
        "evacuation route" in question_lower
        or "approved route" in question_lower
    )

    if asks_for_evacuation_route and not any(
        phrase in context_lower
        for phrase in [
            "evacuation route",
            "designated evacuation route",
        ]
    ):
        logger.warning(
            "No evidence found for the requested evacuation route"
        )
        return (
            "I couldn't verify the company's approved evacuation route "
            "from the available documents. Consult your organization's "
            "approved emergency plan or safety officer."
        )

    # Safety check: company-specific safety officer contact.
    asks_for_safety_officer_contact = (
        "safety officer" in question_lower
        and (
            "phone" in question_lower
            or "number" in question_lower
        )
    )

    if asks_for_safety_officer_contact and not (
        "safety officer" in context_lower
        and any(
            term in context_lower
            for term in [
                "phone",
                "telephone",
                "contact number",
                "phone number",
                "mobile",
            ]
        )
    ):
        logger.warning(
            "No evidence found for the safety officer's contact"
        )
        return (
            "I couldn't verify your workplace safety officer's phone "
            "number from the available documents. Contact your "
            "organization through its approved directory."
        )

    # Safety check: do not invent specific PPE items.
    asks_about_ppe = any(
        term in question_lower
        for term in [
            "personal protective equipment",
            "what ppe",
            "which ppe",
            "ppe items",
        ]
    )

    if asks_about_ppe:
        ppe_items = [
            "gloves",
            "goggles",
            "respirator",
            "hard hat",
            "safety boots",
        ]

        reference_text = " ".join(
            reference.get("text", "")
            for reference in reference_results
        ).lower()

        if not any(item in reference_text for item in ppe_items):
            logger.info(
                "Available OSHA reference does not specify individual PPE items"
            )
            return (
                "The available OSHA reference does not specify individual "
                "PPE items. It mentions fall protection, rescue, "
                "air-monitoring, ventilation, lighting, and communication "
                "equipment. Check your organization's approved SOP and "
                "hazard assessment for the required PPE."
            )

    # Prepare a direct answer when the retrieved OSHA passage supports it.
    asks_about_communication = any(
        term in question_lower
        for term in [
            "communication",
            "communicate with outside personnel",
            "contact with attendant",
            "two-way radio",
        ]
    )

    communication_answer = None

    if asks_about_communication:
        communication_text = " ".join(
            reference.get("text", "")
            for reference in reference_results
        ).lower()

        if "trained attendant" in communication_text:
            logger.info(
                "Communication question supported by OSHA reference"
            )
            communication_answer = (
                "According to the retrieved OSHA reference, maintain "
                "contact at all times with a trained attendant, either "
                "visually, by phone, or by two-way radio. This helps "
                "the attendant and entry supervisor order an evacuation "
                "and alert trained rescue personnel when needed. "
                "Follow your organization's approved confined-space "
                "entry procedures."
            )

    # Generate an answer only when no supported direct answer is available.
    answer = communication_answer

    if answer is None:
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are SOPilot AI, an assistant for workplace safety "
                    "procedures. Answer using only the supplied context. "
                    "Do not invent steps or add unsupported advice. "
                    "If the context does not answer the question, say so. "
                    "Sample procedures are unverified and must not be "
                    "treated as approved instructions. Public references "
                    "are not a replacement for the employer's approved SOP. "
                    "Give a clear answer and include relevant retrieved "
                    "steps when supported by the context. Keep the answer "
                    "concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Retrieved context:\n{context}\n\n"
                    "Answer based only on the retrieved context."
                ),
            },
        ]

        try:
            inputs = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            input_ids = inputs["input_ids"].to("cpu")
            attention_mask = inputs.get("attention_mask")

            if attention_mask is not None:
                attention_mask = attention_mask.to("cpu")

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=180,
                    do_sample=False,
                    repetition_penalty=1.15,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_tokens = outputs[0][input_ids.shape[-1]:]

            answer = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            ).strip()

            if not answer:
                answer = (
                    "The model could not produce a reliable answer "
                    "from the retrieved context."
                )

        except Exception:
            logger.exception("Model answer generation failed")
            answer = (
                "I couldn't generate an answer. Please review the retrieved "
                "documents and consult your organization's approved SOP."
            )

    logger.info(
        "Answer generated; sample_sources=%d, reference_sources=%d",
        len(sample_results),
        len(reference_results),
    )

    # Build the output, including retrieved supporting passages.
    output_parts = [
        "AI answer:",
        answer,
        "",
        "Retrieved information:",
        "",
    ]

    for procedure, score in sample_results:
        output_parts.append(
            f"Procedure: {procedure.get('title', 'Unknown')}"
        )
        output_parts.append(
            f"Procedure ID: {procedure.get('procedureId', 'Unknown')}"
        )
        output_parts.append(
            f"Category: {procedure.get('category', 'Unknown')}"
        )
        output_parts.append("Sample steps:")

        steps = procedure.get("steps", [])

        if isinstance(steps, list):
            for i, step in enumerate(steps):
                if isinstance(step, dict):
                    number = step.get("stepNumber", i + 1)
                    description = step.get("description", "")
                    output_parts.append(f"{number}. {description}")
                else:
                    output_parts.append(f"{i + 1}. {step}")
        else:
            output_parts.append(str(steps))

        output_parts.append("")

    for reference in reference_results:
        output_parts.append(
            f"Source: {reference.get('source_file', 'Unknown')}"
        )
        output_parts.append(
            f"Page: {reference.get('page', 'Unknown')}"
        )
        output_parts.append("Reference excerpt:")
        output_parts.append(reference.get("text", ""))
        output_parts.append("")

    output_parts.append(
        f"Retrieved sources: {len(sample_results)} sample procedure(s), "
        f"{len(reference_results)} public-reference passage(s)."
    )
    output_parts.append(
        "Verify all information against approved workplace procedures "
        "before using it for safety-critical work."
    )

    return "\n".join(output_parts)


if __name__ == "__main__":
    print("SOPilot AI - Fine-Tuned Model + Semantic Retrieval")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("SOPilot AI: Goodbye!")
            break

        if not question:
            continue

        print("\nSOPilot AI:")
        print(ask_sopilot(question))
        print()
