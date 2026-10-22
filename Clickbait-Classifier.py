import io
import os
import sys
import math
import torch
import requests
import numpy as np
import pandas as pd
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaPreTrainedModel, RobertaModel, RobertaConfig, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_curve, auc, average_precision_score
from sklearn.model_selection import train_test_split

# =====================================================================
# 1. GLOBAL SETTINGS & HYPERPARAMETERS
# =====================================================================
model_name = 'roberta-base'  # Hugging Face backbone model
num_labels = 2               # Clickbait (1) vs. Not Clickbait (0)
max_seq_length = 128         # Tokenized padding length
train_batch_size = 32
test_batch_size = 32
num_train_epochs = 3
gradient_accumulation_steps = 1
weight_decay = 0.01
learning_rate = 2e-5
adam_epsilon = 1e-8
warmup_ratio = 0.1

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using execution device: {device}")

# =====================================================================
# 2. FETCH AND CLEAN DATASET VIA PANDAS
# =====================================================================
print("\nFetching dataset from GitHub...")
# Clean URL without volatile security tokens
url = 'https://raw.githubusercontent.com/aaravdaftary/Attention-Bias-Neuron-Comparison/refs/heads/main/Engagement%20Data%20-%20clickbait_data.csv'

# Load dataset once
df = pd.read_csv(url)

# Strip hidden whitespace characters from headers
df.columns = df.columns.str.strip()

text_col = 'headline'
label_col = 'clickbait'

# Split into train and test sets (80% train, 20% test)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[label_col]  # preserves class balance
)

# Unpack inputs into raw structural tuple formats
train_examples = (
    train_df[text_col].astype(str).tolist(),
    train_df[label_col].tolist()
)

test_examples = (
    test_df[text_col].astype(str).tolist(),
    test_df[label_col].tolist()
)

print(f"Train samples: {len(train_examples[0])}")
print(f"Test samples: {len(test_examples[0])}")

# =====================================================================
# 3. INITIALIZE TOKENIZER & WORKAROUND HACKS
# =====================================================================
print("\nInitializing model configuration and tokenizer...")
config_class = RobertaConfig
config = config_class.from_pretrained(model_name, num_labels=num_labels)

tokenizer = AutoTokenizer.from_pretrained(model_name, verbose=False)

# Internal Hugging Face tracking workarounds
with open('dummy.py', 'w') as f:
    f.write("# Dummy file for transformers source check")
sys.modules['__main__'].__file__ = 'dummy.py'

# =====================================================================
# 4. CUSTOM MODEL ARCHITECTURE DEFINITION
# =====================================================================
class RobertaForSmilesClassification(RobertaPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForSmilesClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)

        # Linear projection classification head
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        # Grab the CLS classification token representation from the sequence output
        cls_output = sequence_output[:, 0, :]
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

model_class = RobertaForSmilesClassification
model = model_class.from_pretrained(model_name, config=config).to(device)
print("Model dynamically loaded and structured successfully!")

# =====================================================================
# 5. PYTORCH DATA DATASET WRAPPER
# =====================================================================
class MyClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length=128):
        text, labels = data
        self.examples = tokenizer(text=text, text_pair=None, truncation=True, padding="max_length",
                                  max_length=max_seq_length, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, index):
        return {key: self.examples[key][index] for key in self.examples}, self.labels[index]

# Instantiate dataset classes
train_dataset = MyClassificationDataset(train_examples, tokenizer, max_seq_length)
test_dataset = MyClassificationDataset(test_examples, tokenizer, max_seq_length)

# Establish Datastream Loaders
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=test_batch_size)

# Batched runtime mapping utilities
def get_inputs_dict(batch):
    inputs = {key: value.squeeze(1).to(device) if value.dim() > 2 else value.to(device) for key, value in batch[0].items()}
    inputs["labels"] = batch[1].to(device)
    return inputs

# =====================================================================
# 6. EVALUATION METRICS & HELPERS
# =====================================================================
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def compute_metrics(preds, model_outputs, labels, eval_examples=None):
    assert len(preds) == len(labels)
    mismatched = labels != preds

    text_headlines = eval_examples[0] if isinstance(eval_examples, tuple) else eval_examples
    wrong = [i for (i, v) in zip(text_headlines, mismatched) if v.any()]

    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    scores = np.array([softmax(element)[1] for element in model_outputs])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)
    auprc = average_precision_score(labels, scores)

    return (
        {"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "auroc": auroc, "auprc": auprc},
        wrong,
    )

def print_confusion_matrix(result):
    print('confusion matrix:')
    print('            predicted    ')
    print('          0     |     1')
    print('    ----------------------')
    print('   0 | ', format(result['tn'], '5d'), ' | ', format(result['fp'], '5d'))
    print('gt -----------------------')
    print('   1 | ', format(result['fn'], '5d'), ' | ', format(result['tp'], '5d'))
    print('---------------------------------------------------')

# =====================================================================
# 7. OPTIMIZATION SCHEDULER SETUP
# =====================================================================
t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
optimizer_grouped_parameters = []
custom_parameter_names = set()
no_decay = ["bias", "LayerNorm.weight"]

optimizer_grouped_parameters.extend(
    [
        {
            "params": [p for n, p in model.named_parameters() if n not in custom_parameter_names and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in custom_parameter_names and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
)

warmup_steps = math.ceil(t_total * warmup_ratio)
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

# =====================================================================
# 8. TRAINING & EVALUATION LOOP EXECUTION
# =====================================================================
print("\nInitialization complete. Commencing optimization training sequence...")

for epoch in range(num_train_epochs):
    model.train()
    epoch_loss = []

    for batch in train_dataloader:
        batch = get_inputs_dict(batch)
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        model.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]

        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss.append(loss.item())

    # Evaluation cycle at the conclusion of each epoch
    eval_loss = 0.0
    nb_eval_steps = 0
    n_batches = len(test_dataloader)
    preds = np.empty((len(test_dataset), num_labels))
    out_label_ids = np.empty((len(test_dataset)))
    model.eval()

    for i, test_batch in enumerate(test_dataloader):
        with torch.no_grad():
            test_batch = get_inputs_dict(test_batch)
            input_ids = test_batch['input_ids']
            attention_mask = test_batch['attention_mask']
            labels = test_batch['labels']

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1
        start_index = test_batch_size * i
        end_index = start_index + test_batch_size if i != (n_batches - 1) else len(test_dataset)

        preds[start_index:end_index] = logits.detach().cpu().numpy()
        out_label_ids[start_index:end_index] = test_batch["labels"].detach().cpu().numpy()

    eval_loss = eval_loss / nb_eval_steps
    model_outputs = preds
    preds = np.argmax(preds, axis=1)

    result, wrong = compute_metrics(preds, model_outputs, out_label_ids, test_examples)

    print(f'\n================ EPOCH {epoch} RESULTS ================')
    print('Training Average Loss: ', np.mean(epoch_loss))
    print('Testing Average Loss:  ', eval_loss)
    print("Metrics Evaluation:    ", result)
    print_confusion_matrix(result)

  # 1. Define a save directory
save_directory = "./saved_clickbait_model"

# 2. Save the model weights and configuration
print(f"\nSaving model to {save_directory}...")
model.save_pretrained(save_directory)

# 3. Save the tokenizer alongside it (essential for preprocessing text later)
tokenizer.save_pretrained(save_directory)
print("Model and tokenizer saved successfully!")
