from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from pathlib import Path
import torch.nn as nn
import pandas as pd
import requests
import torch
import os

MODELS = [(BertModel, BertTokenizer, 'bert-base-uncased')]

for model_class, tokenizer_class, pretrained_weights in MODELS:

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    bert_model = model_class.from_pretrained(pretrained_weights)

labels_list = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

class linear_model(nn.Module):
    def __init__(self, bert_model, num_labels):
        super().__init__()

        embed_size = bert_model.config.hidden_size
        dropout_prob = bert_model.config.hidden_dropout_prob

        self.bert = bert_model

        self.pre_classifier = nn.Linear(embed_size, embed_size)

        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Linear(embed_size, num_labels)

    def forward(self, x):

        # Get BERT embeddings for input_ids
        with torch.no_grad():
            # (batch_size, seq_len, hidden_size)
            hidden = self.bert(x)[0]

        # (batch_size, hidden_size)
        hidden = hidden[:,0]

        # (batch_size, hidden_size)
        pooled_output = self.pre_classifier(hidden)
        # (batch_size, hidden_size)
        pooled_output = nn.ReLU()(pooled_output)
        # (batch_size, hidden_size)
        pooled_output = self.dropout(pooled_output)
        # (batch_size, hidden_size)
        logits = self.classifier(pooled_output)

        return logits

s3_model_url = 'https://toxic-model.s3.eu-west-2.amazonaws.com/toxic_model.pt'
path = "./model/"

path_to_model = os.path.join(path, 'toxic_model.pt')
if not os.path.exists(path_to_model):
    print("Model weights not found, downloading from S3...")
    os.makedirs(os.path.join(path), exist_ok=True)
    filename = Path(path_to_model)
    r = requests.get(s3_model_url)
    filename.write_bytes(r.content)

model = linear_model(bert_model, len(labels_list))
model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))

def predict(model, tokenizer, text):

    model.eval()

    tokenized = tokenizer.encode(text, add_special_tokens=True)
    tok_tensor = torch.tensor(tokenized)
    tok_tensor = tok_tensor.unsqueeze(0)
    logits = model(tok_tensor)
    pred = torch.sigmoid(logits)
    pred = pred.detach().cpu().numpy()
    
    result_df = pd.DataFrame(pred, columns=["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"])
    results = result_df.to_dict("record")
    results_list = [sorted(x.items(), key=lambda kv: kv[1], reverse=True) for x in results][0]

    return text, results_list
