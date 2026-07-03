import os
import gc
import json
import torch
import numpy as np
from copy import deepcopy
import torch.nn as nn
from transformers import BertModel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers import AutoTokenizer
from transformers import TrainingArguments

from tqdm.notebook import tqdm

# # Merge data

# iterate the json file in order (priority: High -> Low)
added_uuids= set()
conflict_count = 0
tagged_data = []


# The order is super important
all_datasource = [
    '/data/vm_train_data/top_vendor_hard_cases/',
    '/data/vm_train_data/negative/',
    # ## and more ...
    '/data/vm_train_data/andrew_dataset_partition/main_train_simply_converted.json',
    '/data/vm_train_data/andrew_dataset_partition/main_test_simply_converted.json',
    ## ...
    
]

for ds in all_datasource:
    for json_file in get_json_paths(ds):
        print(f'Loading {json_file}...')
        tmp_json = json.load(open(json_file))
        for d in tqdm(tmp_json):
            uuid = d.get('uuid', '')
            if uuid and uuid in added_uuids:
                conflict_count += 1
                continue
            else:
                if not d['items_label']:
                    d['items_label'] = [['', '']]
                tagged_data.append(d)
                added_uuids.add(uuid)
        print('Current size: {};  Acc conflict: {}'.format(len(tagged_data), conflict_count))
        print()

# ## Build the map:   id <=> name

from collections import Counter
platform_counter = Counter([d['reseller_label'] for d in tagged_data])
platform_counter.most_common(10)

tmp_list = []
for d in tagged_data:
    for item in d['items_label']:
        tmp_list.append(tuple(item))
product_counter = Counter(tmp_list)
del tmp_list
gc.collect()

platform_count_thres = 1
platform_name_2_id = {}
product_count_thres = 10
product_name_2_id = {}

platform_name_2_id[''] = 0

# build maps
i = 1
for k, v in platform_counter.most_common():
    if v >= platform_count_thres and k != '':
        platform_name_2_id[k] = i
        i += 1
platform_id_2_name = {v:k for k, v in platform_name_2_id.items()}


product_name_2_id[('', '')] = i
i += 1
for k, v in product_counter.most_common():
    if v >= product_count_thres and k != ('', ''):
        product_name_2_id[k] = i
        i += 1
product_id_2_name = {v:k for k, v in product_name_2_id.items()}

platform_class_num = len(platform_id_2_name)
product_class_num = len(product_id_2_name)
print(platform_class_num)
print(product_class_num)

id2label = {}
for k, v in platform_id_2_name.items():
    id2label[k] = v
for k, v in product_id_2_name.items():
    id2label[k] = v

label2id = {v:k for k,v in id2label.items()}

from datasets import Dataset

def gen_text(transaction_dict):
    return 'payee: {} ### note: {} \t {}'.format(transaction_dict['vendor_name'], transaction_dict['memo'], transaction_dict['memo_2'])

texts = []
platform_label_list = []
product_label_list = []
merge_label_list = []

# platform_sparse_label_list = []
# product_sparse_label_list = []
sparse_label_list = []

for d in tqdm(tagged_data):
    text = gen_text(d)
    platform_label = []
    texts.append(text)

    platform_sparse_label = []
    i = platform_name_2_id.get(d['reseller_label'], None)
    if i != None:
        platform_sparse_label.append(i)
    if len(platform_sparse_label) == 0:
        platform_sparse_label = [platform_name_2_id['']]

    product_sparse_label = []
    for item in d.get('items_label', []):
        i = product_name_2_id.get(tuple(item), None)
        if i != None:
            product_sparse_label.append(i)
    if len(product_sparse_label) == 0:
        product_sparse_label = [product_name_2_id[('', '')]]

    sparse_label_list.append([platform_sparse_label, product_sparse_label])

# del tagged_data
# gc.collect()

sparse_label_list[:10]

pos_texts = []
neg_texts = []

pos_labels = []
neg_labels = []

for (text, sparse_label) in zip(texts, sparse_label_list):
    platform_sparse_label, product_sparse_label = sparse_label
    if platform_sparse_label[0] == platform_name_2_id[''] and product_sparse_label[0] == product_name_2_id[('','')]:
        neg_texts.append(text)
        neg_labels.append(sparse_label)
    else:
        pos_texts.append(text)
        pos_labels.append(sparse_label)

print(len(pos_texts))
print(len(neg_texts))

# balanced_texts = pos_texts + neg_texts[:len(pos_texts)]
# balanced_labels = pos_labels + neg_labels[:len(pos_labels)]

balanced_texts = pos_texts + neg_texts[:len(pos_texts)]
balanced_labels = pos_labels + neg_labels[:len(pos_labels)]

print(len(balanced_texts))

orig_dataset = Dataset.from_dict({
    'text': balanced_texts,
    'labels': balanced_labels,
})
orig_dataset = orig_dataset.train_test_split(test_size=0.1)
orig_dataset

from transformers import AutoTokenizer
import numpy as np
import torch
import torch.nn.functional as F

model_name = 'distilbert/distilroberta-base'
max_length = 128
tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess_data(examples):
    # take a batch of texts
    text = examples["text"]
    # encode them
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=max_length)
    # # copy labels
    # encoding["labels"] = examples["labels"]

    concat_labels_list = []
    for platform_labels, product_labels in examples["labels"]:
        concat_labels = []
        concat_labels.extend(platform_labels)
        concat_labels.extend(product_labels)
        padded_x = concat_labels + [-1] * (5 - len(concat_labels))
        concat_labels_list.append(padded_x)

    encoding["labels"] = concat_labels_list
    return encoding

encoded_dataset = orig_dataset.map(preprocess_data, batched=True, remove_columns=orig_dataset['train'].column_names)
# encoded_dataset = orig_dataset.map(preprocess_data, batched=True)
encoded_dataset

encoded_dataset['test']['labels'][:10]

# encoded_dataset.set_format("torch")

from transformers import AutoModelForSequenceClassification, RobertaForSequenceClassification
import torch
import torch.nn as nn

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput


# pos_weights = torch.ones(platform_class_num + product_class_num) * 2.0
# pos_weights[0] = 1.0
# pos_weights[platform_class_num] = 1.0
# pos_weights = pos_weights.to('cuda:0')
# print(pos_weights)


class CustomModelForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # loss_fct = BCEWithLogitsLoss(pos_weight=pos_weights)
                loss_fct = BCEWithLogitsLoss()
                multi_hot_labels = torch.zeros_like(logits)
                for i in range(len(labels)):
                    for v in labels[i]:
                        if v != -1:
                            multi_hot_labels[i][v] = 1.0
                        else:
                            break
                # print(logits)
                # print(logits.shape)
                # print(labels)
                # print(labels.shape)
                # print(multi_hot_labels)
                multi_hot_labels = multi_hot_labels.to(logits.device)
                # print('logits', logits)
                # print('multi_hot_labels', multi_hot_labels)
                loss = loss_fct(logits, multi_hot_labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

model = CustomModelForSequenceClassification.from_pretrained(
    model_name,
    problem_type="multi_label_classification", 
    num_labels=len(id2label),
)
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    "/data/trained_models/new_vm_model_1217",
    eval_strategy = "steps",
    save_strategy = "steps",
    logging_steps = 20000,
    save_steps = 20000,
    learning_rate=2e-4,
    # lr_scheduler_type="cosine",  # Learning rate scheduler
    # warmup_steps=10000,  # Number of warmup steps
    per_device_train_batch_size=32,
    per_device_eval_batch_size=128,
    num_train_epochs=8,
    weight_decay=0.01,
    # load_best_model_at_end=False,
    # metric_for_best_model="eval_product_f1",    
    # push_to_hub=True,
)

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from transformers import EvalPrediction
import torch
    
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros_like(predictions)
    y_pred[np.where(probs >= threshold)] = 1

    y_true = np.zeros_like(y_pred)
    for i in range(len(labels)):
        for v in labels[i]:
            if v != -1:
                y_true[i, v] = 1
            else:
                break

    # precision_micro = precision_score(y_true, y_pred, average="micro")
    # recall_micro = recall_score(y_true, y_pred, average="micro")
    # f1_micro = f1_score(y_true, y_pred, average="micro")
    # accuracy = accuracy_score(y_true, y_pred)

    # metrics = {
    #     'precistion': precision_micro,
    #     'recall': recall_micro,
    #     'f1': f1_micro,
    #     'accuracy': accuracy,
    # }

    y_pred_platform = y_pred[:, :platform_class_num]
    y_true_platform = y_true[:, :platform_class_num]
    y_pred_product = y_pred[:, platform_class_num:]
    y_true_product = y_true[:, platform_class_num:]

    platform_precision_micro = precision_score(y_true_platform, y_pred_platform, average="micro")
    platform_recall_micro = recall_score(y_true_platform, y_pred_platform, average="micro")
    platform_f1_micro = f1_score(y_true_platform, y_pred_platform, average="micro")
    platform_acc_micrio = accuracy_score(y_true_platform, y_pred_platform)
    
    product_precision_micro = precision_score(y_true_product, y_pred_product, average="micro")
    product_recall_micro = recall_score(y_true_product, y_pred_product, average="micro")
    product_f1_micro = f1_score(y_true_product, y_pred_product, average="micro")
    product_acc_micro = accuracy_score(y_true_product, y_pred_product)

    metrics = {
        'platform_precistion': platform_precision_micro,
        'platform_recall': platform_recall_micro,
        'platform_f1': platform_f1_micro,
        'platform_accuracy': platform_acc_micrio,
        
        'product_precistion': product_precision_micro,
        'product_recall': product_recall_micro,
        'product_f1': product_f1_micro,
        'product_accuracy': product_acc_micro,
    }
    
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

with open('/data/trained_models/new_vm_model_1217/id2label.json', 'w') as f:
 json.dump(id2label, f)

with open('/data/trained_models/new_vm_model_1217/trainer_log_history.json', 'w') as f:
 json.dump(trainer.state.log_history, f)

trainer.state.log_history[:10]

test_tnxs = [
    {
        'vendor_name': 'V391 Microsoft Office / Azure',
        'memo': 'this is a test',
        'memo_2': 'this is a name'
    },
    {
        'vendor_name': 'Microsoft Azure',
        'memo': 'monthly subscription azure - visual studio enterprise march 2022||abc123456',
        'memo_2': 'Monthly Subscription Azure - Visual Studio Enterprise March 2022',
    },
    {
        'vendor_name': 'Microsoft',
        'memo': 'Office 365',
        'memo_2': 'Office 365',
    },
]

inputs = [gen_text(tnx) for tnx in test_tnxs]
tokens = tokenizer(inputs, padding=True, truncation=True, max_length=max_length)
input_ids = torch.tensor(tokens['input_ids']).to(device)
attention_mask = torch.tensor(tokens['attention_mask']).to(device)
pred = model(input_ids, attention_mask)

for tnx, x in zip(test_tnxs, pred.logits):
    tmp = x.cpu().detach()
    platform_logits = tmp[:platform_class_num]
    product_logits = tmp[platform_class_num:]
    platform_labels = []
    product_labels = []
    for i in np.where(platform_logits > 0)[0]:
        platform_labels.append(platform_id_2_name[i])
    for i in np.where(product_logits > 0)[0]:
        product_labels.append(product_id_2_name[i + platform_class_num])
    print(product_logits)
    print(product_labels)
    print('<Input> ', tnx)
    print('<Prediction>')
    print('  reseller: ', platform_labels)
    print('  product: ', product_labels)
    print()

import pandas as pd
train_log_df = pd.DataFrame([x for x in trainer.state.log_history if 'eval_runtime' in x])
train_log_df

tra
