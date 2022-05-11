import os
import sched
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm

import sampy

def load_data(config):
    data_path = os.path.join(config.data_dir, config.subj + '.json')
    datas = sampy.LoadJson(data_path)[0]
    knowledegs = datas['knowledge1']
    questions = datas['question']
    config.label_num = len(knowledegs)

    config.label2id = {knowledeg: i for i, knowledeg in enumerate(list(knowledegs))}
    train_text, train_label = [], []
    test_text, test_label = [], []
    for knowledeg, q_ids in knowledegs.items():
        label_id = config.label2id[knowledeg]
        text, label = [], []
        for q_id in q_ids:
            question = questions[str(q_id)]
            question.pop('parent')
            question.pop('ans')
            text.append(question)
            label.append(label_id)
        split_id = int(len(text) * 0.9)
        train_text.extend(text[:split_id])
        train_label.extend(label[:split_id])
        test_text.extend(text[:split_id])
        test_label.extend(label[:split_id])

    return train_text, train_label, test_text, test_label


class CLassifyDataset(Dataset):
    def __init__(self, text, label, config):
        super(CLassifyDataset, self).__init__()
        self.config = config
        self.text = text
        self.label = label
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        text = self.text[index]
        label = self.label[index]
        
        encoded = self.tokenizer.encode_plus(text['stem'], pad_to_max_length = True,
                                                        max_length = self.config.max_len, return_tensors = 'pt')
        for k, v in encoded.items():
            encoded[k] = v.squeeze()

        return {**encoded, 'labels': torch.tensor(label, dtype=torch.long)}

def train(data_loader, model, loss, optimizer, device, scheduler=None):
    model.train()
    final_loss = 0.0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        label = data.pop('labels')
        outputs = model(**data)
        ls = loss(outputs, label)
        
        optimizer.zero_grad()
        ls.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        final_loss += ls.item()

    return final_loss / len(data_loader)


def accuracy(data_loader, model, device, threshold=0.5):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items():
                data[k] = v.to(device)

            labels = data.pop('labels')
            outputs = model(**data)
            pred = torch.argmax(outputs, dim=-1)
            correct += torch.sum(labels > pred)
            total += len(labels)
    
    return correct / total

