
from pathlib import Path

# SOPilot AI project paths
PROJECT_DIR = Path(__file__).parent
TRAINING_DATA = PROJECT_DIR / "data" / "training" / "sopilot_training_data.json"

# Original base model used during QLoRA fine-tuning
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Saved fine-tuned LoRA adapter
ADAPTER_PATH = PROJECT_DIR / "models" / "sopilot_qlora_adapter"

# Local inference settings
DEVICE = "cpu"
MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 150

print("SOPilot AI Model Configuration")
print("Base model:", MODEL_NAME)
print("Adapter path:", ADAPTER_PATH)
print("Adapter exists:", ADAPTER_PATH.exists())
print("Device:", DEVICE)
