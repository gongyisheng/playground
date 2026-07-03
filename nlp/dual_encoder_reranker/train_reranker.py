# # Dual Encoder + Reranker

# ### Reranker training

from datetime import datetime
import logging

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderModelCardData,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.util import mine_hard_negatives

import torch

# load dataset
train_dataset = load_dataset("sentence-transformers/natural-questions", split="train")
eval_dataset = load_dataset("sentence-transformers/natural-questions", split="validation")
print(train_dataset)

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# You can specify any Hugging Face pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = "answerdotai/ModernBERT-base"
train_batch_size = 8
num_epochs = 1
num_hard_negatives = 5  # How many hard negatives should be mined for each question-answer pair
output_dir = (
    "/data/yisheng/reranker_2025_05_13/" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

# Load a model to train/finetune
model = CrossEncoder(model_name) # num_labels=1 is for rerankers
print("Model max length:", model.max_length)
print("Model num labels:", model.num_labels)

# mine hard negatives
# The success of training reranker models often depends on the quality of the negatives
embedding_model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")
hard_train_dataset = mine_hard_negatives(
    train_dataset,
    embedding_model,
    num_negatives=num_hard_negatives,  # How many negatives per question-answer pair
    range_min=0,  # Skip the x most similar samples
    range_max=20,  # Consider only the x most similar samples
    margin=0,  # Similarity between query and negative samples should be x lower than query-positive similarity
    sampling_strategy="top",  # Randomly sample negatives from the range
    batch_size=4096,  # Use a batch size of 4096 for the embedding model
    output_format="labeled-pair",  # The output format is (query, passage, label), as required by BinaryCrossEntropyLoss
    use_faiss=False,  # Using FAISS is recommended to keep memory usage low (pip install faiss-gpu or pip install faiss-cpu)
)
print(hard_train_dataset)
print(hard_train_dataset[1])

# loss function
from datasets import load_dataset
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.losses import CachedMultipleNegativesRankingLoss

# Load a model to train/finetune
model = CrossEncoder("xlm-roberta-base", num_labels=1) # num_labels=1 is for rerankers

# Initialize the CachedMultipleNegativesRankingLoss, which requires pairs of
# related texts or triplets
loss = CachedMultipleNegativesRankingLoss(model)

# Load an example training dataset that works with our loss function:
train_dataset = load_dataset("sentence-transformers/gooaq", split="train")

# train
from sentence_transformers.cross_encoder import CrossEncoderTrainingArguments

args = CrossEncoderTrainingArguments(
    # Required parameter:
    output_dir="models/reranker-MiniLM-msmarco-v1",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="reranker-MiniLM-msmarco-v1",  # Will be used in W&B if `wandb` is installed
)

# eval
from datasets import load_dataset
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderCorrelationEvaluator

# Load a model
model = CrossEncoder("cross-encoder/stsb-TinyBERT-L4")

# Load the STSB dataset (https://huggingface.co/datasets/sentence-transformers/stsb)
eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
pairs = list(zip(eval_dataset["sentence1"], eval_dataset["sentence2"]))

# Initialize the evaluator
dev_evaluator = CrossEncoderCorrelationEvaluator(
    sentence_pairs=pairs,
    scores=eval_dataset["score"],
    name="sts_dev",
)
