from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        # Load the pretrained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        # Linear layer for binary classification
        self.linear = nn.Linear(768, 3)

    def forward(self, input_id, mask):
        # Pass the inputs through the BERT model
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        # Pass through dropout layer
        dropout_output = self.dropout(pooled_output)
        # Pass through linear layer
        logits = self.linear(dropout_output)
        return logits
    