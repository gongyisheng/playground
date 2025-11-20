from trl import SFTTrainer

from model import load_model
from dataset import load_training_dataset
from config import get_training_config

# Load model, dataset, and config
model = load_model()
dataset = load_training_dataset()
training_args = get_training_config()

# Create and run trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()
