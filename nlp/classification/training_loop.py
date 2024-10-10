import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from transformers import logging
from config import CFG
from loss import get_loss_artifacts
from utils import set_device
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import mlflow_utils
from tqdm.auto import tqdm


# get rid of the 'Some weights of the model checkpoint' warnings from HF
logging.set_verbosity_error()


def train_model(
    model,
    optimizer,
    lr_scheduler,
    label_encoder: LabelEncoder,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    device: torch.device,
    CFG: CFG,
):
    """Training loop over epochs and batches"""

    # get loss artifacts: (loss_fn, miner_fn) or (loss_fn, loss_optimizer)
    if CFG.loss == "tripletmargin":
        loss_fn, miner_fn = get_loss_artifacts(label_encoder, model, device, CFG)
    elif CFG.loss in ["arcface", "proxyanchor"]:
        loss_fn, loss_optimizer = get_loss_artifacts(label_encoder, model, device, CFG)

    # initialize accuracy calculators
    metrics_device = set_device(CFG.metric_calc_device)
    accuracy_calculators = get_accuracy_calculators(metrics_device, CFG)

    # initialize results list
    results = []

    # Gradient accum alert
    if CFG.gradient_accumulation_steps > 1:
        print(
            f"Using gradient accumulation with number steps = {CFG.gradient_accumulation_steps}"
        )
    # loop epochs
    print(f"Starting training loop")
    for epoch in range(CFG.n_epochs):

        # Training
        loss_train = 0.0
        model.train()
        # loop batches
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"epoch {epoch} train")):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels", None)
            # forward pass; return pooled embeddings
            embeddings = model(**batch)
            # loss-specific components
            if CFG.loss == "tripletmargin":
                indices_tuple = miner_fn(embeddings, labels)
                loss = loss_fn(embeddings, labels, indices_tuple)
                loss.backward()
                # accumulating loss
                loss_train += loss.item()
                # gradient accumulation, optimizer step, zero grad
                if ((step + 1) % CFG.gradient_accumulation_steps == 0) or (
                    step + 1 == len(train_dataloader)
                ):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            elif CFG.loss in ["arcface", "proxyanchor"]:
                loss = loss_fn(embeddings, labels)
                loss.backward()
                # accumulating loss
                loss_train += loss.item()
                # gradient accumulation, optimizer step, zero grad
                if ((step + 1) % CFG.gradient_accumulation_steps == 0) or (
                    step + 1 == len(train_dataloader)
                ):
                    optimizer.step()
                    loss_optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    loss_optimizer.zero_grad()
        epoch_loss_train = loss_train / len(train_dataloader)
        if CFG.mlflow:
            mlflow_utils.log_metric("loss_train", epoch_loss_train, step=epoch)

        # Validation
        ## get embeddings and labels from train sets

        ## get embeddings, labels, and validation loss from eval sets
        if CFG.loss == "tripletmargin":
            embeddings_eval, labels_eval, loss_valid = get_embeddings_labels_loss_req_miner(model, device, eval_dataloader, loss_fn, miner_fn, epoch, CFG)
        elif CFG.loss in ["arcface", "proxyanchor"]:
            embeddings_eval, labels_eval, loss_valid = get_embeddings_labels_loss(model, device, eval_dataloader, loss_fn, epoch, CFG)
        epoch_loss_valid = loss_valid / len(eval_dataloader)
        if CFG.mlflow:
            mlflow_utils.log_metric("loss_valid", epoch_loss_valid, step=epoch)

        # Record epoch results
        epoch_results = {
            "epoch": epoch,
            "loss_train": epoch_loss_train,
            "loss_valid": epoch_loss_valid
        }
            
        # Calculate metrics
        embeddings_train = None # Dummy 
        if CFG.eval_metrics:
            embeddings_train, labels_train = get_embeddings_and_labels(
                model, device, train_dataloader, CFG
            )
            metrics = calculate_metrics(
                accuracy_calculators,
                embeddings_train,
                embeddings_eval,
                labels_eval,
                labels_train,
            )

            epoch_results.update(metrics)
            if CFG.mlflow:
                mlflow_utils.log_metrics(metrics, step=epoch)

        results.append(epoch_results)

        # Print epoch-level training progress
        print(epoch_results)

    return model, results, embeddings_eval, embeddings_train


def get_embeddings_and_labels(model, device, dataloader, CFG:CFG):
    model.eval()
    for i, batch in enumerate(tqdm(dataloader, desc="generating train embeddings")):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels", None)
        with torch.no_grad():
            model_output = model(**batch)
        if i == 0:
            embeddings = model_output.clone()
            labels_out = labels.clone()
        else:
            embeddings = torch.cat((embeddings, model_output), 0)
            labels_out = torch.cat((labels_out, labels), 0)
    if CFG.metric_calc_device == "cpu":
        embeddings = embeddings.cpu().numpy()
        labels_out = labels_out.cpu().numpy()
    return embeddings, labels_out

def get_embeddings_labels_loss_req_miner(model, device, dataloader, loss_fn, miner_fn, epoch, CFG:CFG):
    model.eval()
    loss_valid = 0.0
    for i, batch in enumerate(tqdm(dataloader, desc=f"epoch {epoch} valid")):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels", None)
        with torch.no_grad():
            model_output = model(**batch)
            indices_tuple = miner_fn(model_output, labels)
            loss = loss_fn(model_output, labels, indices_tuple)
            loss_valid += loss.item()
        if CFG.eval_metrics:
            if i == 0:
                embeddings = model_output.clone()
                labels_out = labels.clone()
            else:
                embeddings = torch.cat((embeddings, model_output), 0)
                labels_out = torch.cat((labels_out, labels), 0)
        else:
            embeddings = None 
            labels_out = None
    if CFG.eval_metrics and CFG.metric_calc_device == "cpu":
        embeddings = embeddings.cpu().numpy()
        labels_out = labels_out.cpu().numpy()
    return embeddings, labels_out, loss_valid

def get_embeddings_labels_loss(model, device, dataloader, loss_fn, epoch, CFG:CFG):
    model.eval()
    loss_valid = 0.0
    for i, batch in enumerate(tqdm(dataloader, desc=f"epoch {epoch} eval")):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels", None)
        with torch.no_grad():
            model_output = model(**batch)
            loss = loss_fn(model_output, labels)
            loss_valid += loss.item()
        if CFG.eval_metrics:
            if i == 0:
                embeddings = model_output.clone()
                labels_out = labels.clone()
            else:
                embeddings = torch.cat((embeddings, model_output), 0)
                labels_out = torch.cat((labels_out, labels), 0)
        else:
            embeddings = None
            labels_out = None
    if CFG.eval_metrics and CFG.metric_calc_device == "cpu":
        embeddings = embeddings.cpu().numpy()
        labels_out = labels_out.cpu().numpy()
    return embeddings, labels_out, loss_valid


def get_accuracy_calculators(metrics_device, CFG: CFG):
    accuracy_calculators = []

    if "map@r" in CFG.eval_metrics:
        accuracy_calculators.append(
            AccuracyCalculator(
                include=("mean_average_precision_at_r",),
                k="max_bin_count",
                device=metrics_device,
            )
        )
    if set(["ami", "precision@1"]).issubset(set(CFG.eval_metrics)):
        accuracy_calculators.append(
            AccuracyCalculator(
                include=("precision_at_1", "AMI"), k=1, device=metrics_device
            )
        )
    elif "ami" in CFG.eval_metrics:
        accuracy_calculators.append(
            AccuracyCalculator(
            include=("AMI",), k=1, device=metrics_device
            )
        )
    elif "precision@1" in CFG.eval_metrics:
        accuracy_calculators.append(
            AccuracyCalculator(
                include=("precision_at_1",), k=1, device=metrics_device
            )
        )

    return accuracy_calculators


def calculate_metrics(
    accuracy_calculators, embeddings_train, embeddings_eval, labels_eval, labels_train
):

    metrics = {}
    for accuracy_calculator in accuracy_calculators:
        metrics.update(
            accuracy_calculator.get_accuracy(
                query=embeddings_eval,
                reference=embeddings_train,
                query_labels=labels_eval,
                reference_labels=labels_train,
                ref_includes_query=False,
            )
        )

    return metrics
