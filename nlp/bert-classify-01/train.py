from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import pandas as pd
import evaluate

DATA_DIR = "/media/hdddisk/bert-classify-smsspam-data"

# Train the model
import mlflow
mlflow.set_tracking_uri("https://mlflow.yellowday.day")
mlflow.set_experiment("bert-classify-smsspam")
mlflow.enable_system_metrics_logging()  # Logs CPU, RAM, GPU usage

df = pd.read_csv("./data/smsspam.csv")

# split into train and test
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

train_df.to_csv("./data/smsspam-train.csv", index=False)
test_df.to_csv("./data/smsspam-test.csv", index=False)

print("Train size:", len(train_df), "Test size:", len(test_df))

# Load the dataset
dataset = load_dataset("csv", data_files={"train": "./data/smsspam-train.csv", "test": "./data/smsspam-test.csv"})
mlflow.log_artifact("/home/yisheng/playground/nlp/bert-classify-smsspam/data/smsspam-train.csv")
mlflow.log_artifact("./data/smsspam-test.csv")

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare the dataset for PyTorch
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Load the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir=f"{DATA_DIR}/results",  # Output directory
    num_train_epochs=1,                # Number of training epochs
    per_device_train_batch_size=8,     # Batch size for training
    per_device_eval_batch_size=8,      # Batch size for evaluation
    learning_rate=5e-6,                # Learning rate
    warmup_steps=100,                  # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,                 # Strength of weight decay
    logging_dir=f"{DATA_DIR}/logs",    # Directory for storing logs
    logging_steps=10,
    eval_strategy="steps",             # Evaluate every epoch = "epoch"
    eval_steps=20,                     # Number of steps between evaluations
    save_strategy="steps",             # Save model every epoch
    save_steps=20,
    load_best_model_at_end=True,       # Load the best model at the end of training
    save_total_limit=2,                # Limit the total number of saved models
    metric_for_best_model="accuracy",  # Use accuracy to determine the best model
    report_to="mlflow",                # Enable logging to MLflow
)

# Define the metric for evaluation
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Initialize the Trainer
trainer = Trainer(
    model=model,                                 # The pre-trained model
    args=training_args,                          # Training arguments
    train_dataset=tokenized_datasets["train"],   # Training dataset
    eval_dataset=tokenized_datasets["test"],     # Evaluation dataset
    compute_metrics=compute_metrics,             # Function to compute metrics
)

with mlflow.start_run():
    trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save the model
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")

print("Model training and evaluation complete. Model saved to './final_model'.")

# Resume from checkpoint
dataset = load_dataset("csv", data_files={"train": "./data/smsspam-train.csv", "test": "./data/smsspam-test.csv"})


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("/media/hdddisk/bert-classify-smsspam-data/results/checkpoint-460", num_labels=2)

training_args = TrainingArguments(
    output_dir=f"{DATA_DIR}/results",  # Output directory
    num_train_epochs=2,                # Number of training epochs
    per_device_train_batch_size=8,     # Batch size for training
    per_device_eval_batch_size=8,      # Batch size for evaluation
    learning_rate=5e-6,                # Learning rate
    warmup_steps=100,                  # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,                 # Strength of weight decay
    logging_dir=f"{DATA_DIR}/logs",    # Directory for storing logs
    logging_steps=10,
    eval_strategy="steps",             # Evaluate every epoch = "epoch"
    eval_steps=20,                     # Number of steps between evaluations
    save_strategy="steps",             # Save model every epoch
    save_steps=20,
    load_best_model_at_end=True,       # Load the best model at the end of training
    save_total_limit=2,                # Limit the total number of saved models
    metric_for_best_model="accuracy",  # Use accuracy to determine the best model
    report_to="mlflow",                # Enable logging to MLflow
)

# Define the metric for evaluation
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Initialize the Trainer
trainer = Trainer(
    model=model,                                 # The pre-trained model
    args=training_args,                          # Training arguments
    train_dataset=tokenized_datasets["train"],   # Training dataset
    eval_dataset=tokenized_datasets["test"],     # Evaluation dataset
    compute_metrics=compute_metrics,             # Function to compute metrics
)
