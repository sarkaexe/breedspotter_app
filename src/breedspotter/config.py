from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class TrainConfig(BaseModel):
    model_name: str = os.getenv("MODEL_NAME", "timm_efficientnet_b0")
    num_classes: int = int(os.getenv("NUM_CLASSES", "120"))
    image_size: int = int(os.getenv("IMAGE_SIZE", "224"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "32"))
    epochs: int = int(os.getenv("EPOCHS", "10"))
    lr: float = float(os.getenv("LR", "3e-4"))
    data_dir: str = os.getenv("DATA_DIR", "./data")
    checkpoint_dir: str = os.getenv("CHECKPOINT_DIR", "./checkpoints")
