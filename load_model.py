
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from model_config import MODEL_NAME, ADAPTER_PATH, DEVICE

print("Loading SOPilot AI base model...")
print("Base model:", MODEL_NAME)
print("Adapter path:", ADAPTER_PATH)

if not ADAPTER_PATH.exists():
    raise FileNotFoundError(
        f"Fine-tuned adapter folder not found: {ADAPTER_PATH}"
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

print("Loading fine-tuned LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    str(ADAPTER_PATH)
)

model.to(DEVICE)
model.eval()

print("Base model loaded successfully!")
print("Fine-tuned adapter loaded successfully!")
print("SOPilot AI is ready for inference.")
