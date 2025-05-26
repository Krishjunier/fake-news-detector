import pandas as pd
import numpy as np
import re
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# ✅ Load CSV files with error handling
data_fake = pd.read_csv('Data-set\Fake.csv', engine='python', on_bad_lines='warn')
data_true = pd.read_csv('Data-set\True.csv', engine='python', on_bad_lines='warn')

# ✅ Label and concat
data_fake['label'] = 0
data_true['label'] = 1
df = pd.concat([data_fake, data_true])[['text', 'label']].dropna().reset_index(drop=True)

# ✅ Clean text (keep structure important for BERT)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@\w+|#\w+", '', text)
    text = re.sub(r"\n", ' ', text)
    text = re.sub(r"[^a-zA-Z0-9\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

df['text'] = df['text'].apply(clean_text)

# ✅ Balance dataset
fake = df[df['label'] == 0]
real = df[df['label'] == 1]
min_len = min(len(fake), len(real))
df_balanced = pd.concat([fake.sample(min_len, random_state=42), real.sample(min_len, random_state=42)]).sample(frac=1).reset_index(drop=True)

# ✅ Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df_balanced['text'].tolist(),
    df_balanced['label'].tolist(),
    test_size=0.2,
    random_state=42
)

# ✅ Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# ✅ Dataset class
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, test_labels)

# ✅ Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# ✅ Metrics
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# ✅ Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    report_to='none'
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# ✅ Train and save
trainer.train()
trainer.evaluate()
model.save_pretrained('./fake-news-bert-model')
tokenizer.save_pretrained('./fake-news-bert-model')
