
from transformers import AutoTokenizer, AutoModelForCausalLM
from model_config import MODEL_NAME, MAX_NEW_TOKENS
import torch

print("Loading SOPilot AI for testing...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
)
model.to("cpu")
model.eval()

question = "What steps should be followed before entering a confined space?"

prompt = f"""You are SOPilot AI, a manufacturing safety procedure assistant.
Answer using the provided sample procedure information.
If the information is insufficient, say that the procedure must be checked
against the organization's approved safety instructions.

Question: {question}

Answer:"""

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

print("\nQuestion:", question)
print("\nSOPilot AI response:\n")
print(answer)
