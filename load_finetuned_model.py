import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

PROJECT_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = PROJECT_DIR / "models" / "sopilot_qlora_adapter"

print("Loading SOPilot AI fine-tuned model...")
print("Base model:", BASE_MODEL)
print("Adapter directory:", ADAPTER_DIR)

if not ADAPTER_DIR.exists():
    raise FileNotFoundError(
        f"Fine-tuned adapter not found: {ADAPTER_DIR}"
    )

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float32,
    device_map="cpu"
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR
)

model.eval()

print("Fine-tuned model loaded successfully!")