import torch.nn as nn
from transformers import BertModel


class Classify(nn.Module):
    def __init__(self, config):
        super(Classify, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc = nn.Sequential(nn.Linear(self.bert.config.hidden_size, 256), nn.Sigmoid(), nn.Linear(256, len(config.label2id)))
        # self.fc = nn.Sequential(nn.Linear(self.bert.config.hidden_size, len(config.label2id)))

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs[0][:,0,:]
        return self.fc(self.dropout(cls_output))