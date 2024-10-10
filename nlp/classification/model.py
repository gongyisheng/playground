# Prepare HF model, optimizer, and lr_scheduler in the torch way

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect
from transformers import (
    AutoConfig,
    AutoModel,
    PreTrainedModel,
    PretrainedConfig,
    get_scheduler,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from config import CFG


def setup_model(
    train_dataloader: DataLoader,
    device: torch.device,
    CFG: CFG,
):
    """Prepare HF model torch lr_scheduler"""
    model = build_model(CFG)
    optimizer = get_optimizer(model, CFG)
    num_training_steps = (
        math.ceil(len(train_dataloader) / CFG.gradient_accumulation_steps)
        * CFG.n_epochs
    )
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    model.to(device)
    return model, optimizer, lr_scheduler


def get_optimizer(model, CFG: CFG):
    """Prepare torch optimizer; this will make more sense with selecting from various optimizer options"""
    if CFG.optimizer_name == "AdamW":
        return AdamW(model.parameters(), lr=CFG.lr)
    else:
        raise ValueError(f"Optimizer {CFG.optimizer_name} not supported.")

def build_model(CFG: CFG):
    if CFG.build_custom_model_type == "metric":
        config = SeqEmbedConfig(CFG.pool_mode, CFG.checkpoint)
        return SeqEmbedModel(config)
    elif CFG.build_custom_model_type == "num_metric":
        num_feat_list = get_num_col_names_iter(CFG.num_col_names)
        config = NumSeqConfig(CFG.pool_mode, CFG.checkpoint, num_feat_list)
        return NumSeqModel(config)
    else:
        raise ValueError("Could not build model of specified type.")

class CustomPooling(nn.Module):
    """Custom token pooling module. Supports mean, max, and context pooling."""

    def __init__(self, mode: str = "mean", config=None):
        super(CustomPooling, self).__init__()
        self.mode = mode
        self.config = config

        if config and mode == "context":
            self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)

    def forward(self, last_hidden_state, attention_mask):
        """last_hidden_state = First element of model_output, which contains all token embeddings"""
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        if self.mode == "mean":
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            return mean_embeddings
        elif self.mode == "max":
            last_hidden_state[
                input_mask_expanded == 0
            ] = -1e9  # Set padding tokens to large negative value
            max_embeddings = torch.max(last_hidden_state, 1)[0]
            return max_embeddings
        elif self.mode == "context" and self.config:
            context_token = last_hidden_state[:, 0]
            pooled_output = self.dense(context_token)
            pooled_output = F.gelu(pooled_output)
            return pooled_output


# class SeqEmbedModel(nn.Module):
#     """Custom transformer built on a HF base model. It pools token embeddings together and returns the resulting
#     sequence embeddings."""

#     def __init__(self, checkpoint: str, pool_mode: str = "mean"):
#         super().__init__()
#         self.checkpoint = checkpoint
#         self.pool_mode = pool_mode
#         self.config = AutoConfig.from_pretrained(
#             self.checkpoint, output_hidden_states=True
#         )
#         self.model = AutoModel.from_pretrained(self.checkpoint, config=self.config)
#         self.pool = CustomPooling(mode=self.pool_mode)
#         self.embedding_size = self.config.hidden_size

#     def forward(self, **inputs) -> torch.Tensor:
#         outputs = self.model(**inputs)
#         embeddings = self.pool(outputs["last_hidden_state"], inputs["attention_mask"])
#         return embeddings

class SeqEmbedConfig(PretrainedConfig):
    model_type = "SeqEmbedModel"
    def __init__(self, pool_mode="mean", checkpoint="microsoft/mdeberta-v3-base", **kwargs):
        super().__init__(**kwargs)
        self.pool_mode = pool_mode
        self.checkpoint = checkpoint

    def __repr__(self):
        return self.__class__.__name__

class SeqEmbedModel(PreTrainedModel):
    """Custom transformer built on a HF base model. It pools token embeddings together and returns the resulting
    sequence embeddings."""

    config_class = SeqEmbedConfig

    def __init__(self, config:SeqEmbedConfig):
        super().__init__(config)
        self.config = config
        self.backbone_config = AutoConfig.from_pretrained(
            self.config.checkpoint, output_hidden_states=True
        )
        self.backbone = AutoModel.from_pretrained(self.config.checkpoint, config=self.backbone_config)
        self.pool = CustomPooling(mode=self.config.pool_mode)
        self.embedding_size = self.backbone_config.hidden_size

    def forward(self, **inputs) -> torch.Tensor:
        outputs = self.backbone(**inputs)
        embeddings = self.pool(outputs["last_hidden_state"], inputs["attention_mask"])
        return embeddings

class NumSeqConfig(PretrainedConfig):
    model_type = "NumSeqModel"
    def __init__(self, pool_mode="mean", num_feat_list=None, checkpoint="microsoft/mdeberta-v3-base", **kwargs):
        super().__init__(**kwargs)
        self.pool_mode = pool_mode
        self.num_feat_list = num_feat_list
        self.checkpoint = checkpoint

    def __repr__(self):
        return self.__class__.__name__
    
class NumSeqModel(PreTrainedModel):
    """Custom transformer built on a HF base model. It concatenates numerical features with pooled token embeddings
    and passes them through a feedforward layer."""

    config_class = NumSeqConfig

    def __init__(self, config: NumSeqConfig):
        super().__init__(config)
        self.config = config 

        self.backbone_config = AutoConfig.from_pretrained(
            self.config.checkpoint, output_hidden_states=True
        )
        self.backbone = AutoModel.from_pretrained(self.config.checkpoint, config=self.backbone_config)

        self.pool = CustomPooling(mode=self.config.pool_mode)

        self.ff1 = nn.Sequential(
            nn.Linear(
                self.backbone_config.hidden_size + len(self.config.num_feat_list),
                self.backbone_config.hidden_size + len(self.config.num_feat_list),
            ),
            nn.LayerNorm(self.backbone_config.hidden_size + len(self.config.num_feat_list)),
            nn.ReLU(),
        )

        self.ff2 = nn.Linear(
            self.backbone_config.hidden_size + len(self.config.num_feat_list),
            self.backbone_config.hidden_size + len(self.config.num_feat_list),
        )

    def split_inputs(self, inputs: dict):
        base_args = inspect.getfullargspec(
            self.backbone.forward
        ).args  # Check which kwargs can be passed to the base model's forward() method
        base_inputs = {
            k: v
            for k, v in inputs.items()
            if (k not in self.config.num_feat_list and k in base_args)
        }  # Exclude num feats and include only expected base_args
        num_inputs = {k: v for k, v in inputs.items() if k in self.num_feat_list}
        return base_inputs, self.reshape_num_feats(num_inputs)

    def reshape_num_feats(self, num_inputs: dict):
        return torch.cat(
            [v.unsqueeze(-1) for k, v in sorted(num_inputs.items())], dim=1
        )  # For inference, we need these num feats to be same place in tensor as training num feats. sort list by keys
    
    def get_embeddings(self, base_inputs):
        outputs = self.backbone(
            **base_inputs
        )  # don't pass num feats or unexpected kwargs to base model
        embeddings = self.pool(
            outputs["last_hidden_state"], base_inputs["attention_mask"]
        )
        return embeddings

    def forward(self, **inputs) -> torch.Tensor:
        base_inputs, num_inputs = self.split_inputs(inputs)
        embeddings = self.get_embeddings(**base_inputs)
        embeddings_aug = torch.cat([embeddings, num_inputs], dim=1)
        out = self.ff1(embeddings_aug)
        out = self.ff2(out)
        return out

def get_num_col_names_iter(num_col_names):
    if isinstance(num_col_names, str):
        return [num_col_names]
    elif isinstance(num_col_names, list):
        return num_col_names