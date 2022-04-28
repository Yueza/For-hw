import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from config import Config
from preprocess import read_sen_pairs
from model import SentenceBERT, BertClassifier
from dataset import SentencePairDataset, SingleBertDataset
from train_eval import train, evaluate, accuracy, similarity_accuracy
import os, argparse
from tqdm import tqdm
import pandas as pd

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
    return parser.parse_args()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    args = set_args()
    print(args)
    config = Config(**vars(args))
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=config.logging_file_name,
        filemode='a+',
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    test_sen_a_list, test_sen_b_list, test_labels = read_sen_pairs(config.test_path)

    # Single BERT
    # train_set = SingleBertDataset(train_sen_a_list, train_sen_b_list, train_labels, config)
    # test_set = SingleBertDataset(test_sen_a_list, test_sen_b_list, test_labels, config)

    # SentenceBERT
    test_set = SentencePairDataset(test_sen_a_list, test_sen_b_list, test_labels, config)

    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SentenceBERT(config).to(device)

    best_model_name = '_epoch_3.pth'
    model.load_state_dict(torch.load(best_model_name))

    nodes = pd.read_csv('./data/nodes.tsv', sep='\t')['nodes']
    golden_nodes = pd.read_csv('./data/golden_nodes.tsv', sep='\t')['nodes']

    model.eval()
    scores = []
    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader)):
            for k, v in data.items():
                data[k] = v.to(device)

            label = data.pop('label')
            outputs = model(**data)
            scores.extend(outputs[:, 1])

    correct, total = 0, len(scores) // len(nodes)
    for i, score in enumerate(scores):
        if i == 0:
            temp = []
        elif i % len(nodes) == 0:
            pred_node = nodes[temp.index(max(temp))]
            golden_node = golden_nodes[i // 125 - 1]
            if pred_node == golden_node:
                correct += 1
            temp = []
        temp.append(score)

    acc = correct / total
    print(f"Best Model '{best_model_name}' | Test ACC: {acc}")
    logger.info(f"Best Model '{best_model_name}' | Test ACC: {acc}")





