from trl import SFTTrainer

from model import load_model
from dataset import load_train_dataset
from config import training_args

# Load model, dataset, and config
model = load_model()
dataset = load_train_dataset()

# Create and run trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()
