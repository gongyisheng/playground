# Pytorch Metric Learning Loss

import torch
import torch.nn as nn
from pytorch_metric_learning import losses, miners, distances, reducers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from transformers import PreTrainedModel

from config import CFG
from model import NumSeqModel, SeqEmbedModel


def get_loss_artifacts(
    label_encoder: LabelEncoder, model: PreTrainedModel, device: torch.device, CFG: CFG
):
    """Select a metric learning loss
    Options:
    - tripletmargin = triplet margin loss
    - arcface = arcface loss
    - proxyanchor = proxy anchor loss
    """
    if CFG.loss == "tripletmargin":
        loss_artifacts = get_tripletmargin_artifacts(device, CFG)
    elif CFG.loss == "arcface":
        loss_artifacts = get_arcface_artifacts(label_encoder, model, device, CFG)
    elif CFG.loss == "proxyanchor":
        loss_artifacts = get_proxyanchor_artifacts(label_encoder, model, device, CFG)
    else:
        raise ValueError(f"Loss function {CFG.loss} not supported.")
    return loss_artifacts


def get_arcface_artifacts(
    label_encoder: LabelEncoder, model: PreTrainedModel, device: torch.device, CFG: CFG
):
    loss_kwargs = {
        "num_classes": len(label_encoder.classes_),
        "reducer": get_loss_reducer(CFG),
        "embedding_size": model.backbone_config.hidden_size, #TODO Need to update for numseq model
        "margin": get_loss_margin(CFG),
    }
    loss_kwargs = {k: v for k, v in loss_kwargs.items() if v is not None}

    if loss_kwargs:
        loss_fn = losses.ArcFaceLoss(**loss_kwargs).to(device=device)
    else:
        loss_fn = losses.ArcFaceLoss().to(device=device)

    loss_optimizer = get_loss_optimizer(loss_fn, CFG)

    return loss_fn, loss_optimizer


def get_proxyanchor_artifacts(
    label_encoder: LabelEncoder, model: PreTrainedModel, device: torch.device, CFG: CFG
):
    loss_kwargs = {
        "num_classes": len(label_encoder.classes_),
        "reducer": get_loss_reducer(CFG),
        "distance": get_loss_distance(CFG),
        "embedding_size": model.backbone_config.hidden_size, #TODO Need to update for numseq model
        "margin": get_loss_margin(CFG),
    }
    loss_kwargs = {k: v for k, v in loss_kwargs.items() if v is not None}

    if loss_kwargs:
        loss_fn = losses.ProxyAnchorLoss(**loss_kwargs).to(device=device)
    else:
        loss_fn = losses.ProxyAnchorLoss().to(device=device)

    loss_optimizer = get_loss_optimizer(loss_fn, CFG)

    return loss_fn, loss_optimizer


def get_tripletmargin_artifacts(device: torch.device, CFG: CFG):
    loss_kwargs = {
        "distance": get_loss_distance(CFG),
        "reducer": get_loss_reducer(CFG),
        "margin": get_loss_margin(CFG),
    }
    loss_kwargs = {k: v for k, v in loss_kwargs.items() if v is not None}

    if loss_kwargs:
        loss_fn = losses.TripletMarginLoss(**loss_kwargs).to(device=device)
    else:
        loss_fn = losses.TripletMarginLoss().to(device=device)

    miner_kwargs = {k: v for k, v in loss_kwargs.items() if k in ["margin", "distance"]}
    miner_kwargs.update({"type_of_triplets": get_miner_type_of_triplets(CFG)})
    miner_kwargs = {k: v for k, v in miner_kwargs.items() if v is not None}

    if miner_kwargs:
        miner_fn = miners.TripletMarginMiner(**miner_kwargs)
    else:
        miner_fn = miners.TripletMarginMiner()

    return loss_fn, miner_fn


def get_miner_type_of_triplets(CFG: CFG):
    if CFG.miner_type_of_triplets == "default":
        return None
    else:
        return CFG.miner_type_of_triplets


def get_loss_optimizer(loss_fn, CFG: CFG):
    # TODO: Use lr scheduler?
    if CFG.loss_optimizer == "AdamW":
        return AdamW(loss_fn.parameters(), lr=CFG.loss_lr)
    else:
        raise ValueError(f"Optimizer {CFG.optimizer_name} not supported.")


def get_loss_distance(CFG: CFG):
    # TODO: Implement nondefault p, power, normalize_embeddings
    if CFG.loss_distance == "default":
        return None
    elif CFG.loss_distance == "cosine":
        return distances.CosineSimilarity()
    elif CFG.loss_distance == "dot":
        return distances.DotProductSimilarity()
    elif CFG.loss_distance == "lp":
        return distances.LpDistance()
    elif CFG.loss_distance == "snr":
        return distances.SNRDistance()
    else:
        raise ValueError(f"Loss distance function {CFG.loss_distance} not supported.")


def get_loss_reducer(CFG: CFG):
    # TODO: Implement class weights reducer, peranchor reducer, divisor reducer
    if CFG.loss_reducer == "default":
        return None
    elif CFG.loss_reducer == "avgnonzero":
        return reducers.AvgNonZeroReducer()
    elif CFG.loss_reducer == "mean":
        return reducers.MeanReducer()
    elif CFG.loss_reducer == "sum":
        return reducers.SumReducer()
    else:
        raise ValueError(f"Loss reduction method {CFG.loss_distance} not supported.")


# TODO: Implement Weight Regularizers


def get_loss_margin(CFG: CFG):
    if CFG.loss_margin == "default":
        return None
    else:
        return CFG.loss_margin
