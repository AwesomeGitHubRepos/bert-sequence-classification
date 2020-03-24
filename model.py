from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import torch.nn as nn
import pandas as pd
import torch

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

model = linear_model(bert_model, len(labels_list))
model.load_state_dict(torch.load("./model/toxic_model.pt", map_location=torch.device('cpu')))

def predict(model, tokenizer, text):

    model.eval()

    tokenized = tokenizer.encode(text, add_special_tokens=True)
    tok_tensor = torch.tensor(tokenized)
    tok_tensor = tok_tensor.unsqueeze(0)
    logits = model(tok_tensor)
    pred = torch.sigmoid(logits)
    pred = pred.detach().cpu().numpy()
    
    rounded_pred = [round(x * 100, 1) for x in pred[0]]

    result_df = pd.DataFrame([rounded_pred], columns=["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"])
    results = result_df.to_dict("record")
    results_list = [sorted(x.items(), key=lambda kv: kv[1], reverse=True) for x in results][0]

    return text, results_list
