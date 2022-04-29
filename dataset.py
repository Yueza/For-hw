from multiprocessing import pool
from torch.utils.data import Dataset, DataLoader
from config import Config
import torch
from preprocess import read_sen_pairs

def get_node_resp(model, text, config):
    encoded = config.bert_tokenizer.encode_plus(text, pad_to_max_length = True,
                                                        max_length = config.sen_a_max_len, return_tensors = 'pt')
    for k, v in encoded.items():
            encoded[k] = v.squeeze()
    output = model.bert(
            input_ids=encoded['input_ids'],
            token_type_ids=encoded['token_type_ids'],
            attention_mask=encoded['attention_mask']
        )
    # cls_output = output[0][:,0,:]
    pooling_output = output[0].mean(dim=1)
    return pooling_output
    return cls_output

def get_q_resp(model, text, config):
    encoded = config.bert_tokenizer.encode_plus(text, pad_to_max_length = True,
                                                        max_length = config.sen_b_max_len, return_tensors = 'pt')
    for k, v in encoded.items():
            encoded[k] = v.squeeze()
    output = model.bert(
            input_ids=encoded['input_ids'],
            token_type_ids=encoded['token_type_ids'],
            attention_mask=encoded['attention_mask']
        )
    # cls_output = output[0][:,0,:]
    pooling_output = output[0].mean(dim=1)
    return pooling_output

class SentencePairDataset(Dataset):
    """
    两个句子分别输入BERT, [CLS] sen_a [SEP], [CLS] sen_b [SEP]
    """
    def __init__(self, sen_a_list, sen_b_list, labels, config):
        super(SentencePairDataset, self).__init__()
        self.sen_a_list = sen_a_list
        self.sen_b_list = sen_b_list
        self.labels = labels
        self.config = config

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sen_a = self.sen_a_list[index]
        sen_b = self.sen_b_list[index]
        sen_a_encoded = self.config.bert_tokenizer(sen_a, padding='max_length',
                                                   max_length=self.config.sen_a_max_len, return_tensors='pt')
        sen_b_encoded = self.config.bert_tokenizer(sen_b, padding='max_length',
                                                   max_length=self.config.sen_b_max_len, return_tensors='pt')
        sen_a_encoded['input_ids'], sen_b_encoded['input_ids'] = \
            sen_a_encoded['input_ids'].squeeze(), sen_b_encoded['input_ids'].squeeze()
        sen_a_encoded['token_type_ids'], sen_b_encoded['token_type_ids'] = \
            sen_a_encoded['token_type_ids'].squeeze(), sen_b_encoded['token_type_ids'].squeeze()
        sen_a_encoded['attention_mask'], sen_b_encoded['attention_mask'] = \
            sen_a_encoded['attention_mask'].squeeze(), sen_b_encoded['attention_mask'].squeeze()

        return {
            'sen_a_input_ids': sen_a_encoded['input_ids'],
            'sen_a_token_type_ids': sen_a_encoded['token_type_ids'],
            'sen_a_attention_mask': sen_a_encoded['attention_mask'],
            'sen_b_input_ids': sen_b_encoded['input_ids'],
            'sen_b_token_type_ids': sen_b_encoded['token_type_ids'],
            'sen_b_attention_mask': sen_b_encoded['attention_mask'],
            'label': torch.tensor([self.labels[index]], dtype=torch.long)[0]
        }


if __name__ == '__main__':
    config = Config()
    sen_a_list, sen_b_list, labels = read_sen_pairs(config.test_path)
    print(config.bert_tokenizer(sen_a_list[0], sen_b_list[0], padding='max_length', max_length=config.max_len, return_tensors='pt'))
    # TODO 验证SingleBertDataset
    dataset = SingleBertDataset(sen_b_list, sen_b_list, labels, config)
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    for data in data_loader:
        for k, v in data.items():
            print(k, v)
    # print(dataset[0])
    # print(dataset[0].keys())




