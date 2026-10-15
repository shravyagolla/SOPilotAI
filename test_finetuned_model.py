import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

PROJECT_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = PROJECT_DIR / "models" / "sopilot_qlora_adapter"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu"
)

print("Loading fine-tuned adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

test_questions = [
    "What should be checked before entering a confined space?",
    "Why is atmospheric testing important in a confined space?",
    "What should a worker do if communication with the attendant is lost?"
]

for question in test_questions:
    prompt = (
        "You are SOPilot AI, a manufacturing safety procedure assistant.\n\n"
        f"### Instruction:\n{question}\n\n"
        "### Context:\nUse only verified information from approved safety procedures. "
        "If the required information is unavailable, say so clearly.\n\n"
        "### Response:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("\n" + "=" * 60)
    print("QUESTION:", question)
    print("ANSWER:", answer.strip())

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

print("SOPilot AI answer:")
print(answer)