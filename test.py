import enum
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from config import Config
from preprocess import read_sen_pairs
from model import SentenceBERT, BertClassifier
from dataset import SentencePairDataset, SingleBertDataset, get_node_resp, get_q_resp
from train_eval import train, evaluate, accuracy, similarity_accuracy
import os, argparse
from tqdm import tqdm
import pandas as pd
import sampy

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SentenceBERT(config).to(device)

    best_model_name = '_epoch_3.pth'
    model.load_state_dict(torch.load(best_model_name))
    node_encoder = model.bert
    question_encoder = model.bert

    nodes = pd.read_csv('./data/nodes.tsv', sep='\t')['nodes']
    test_datas = sampy.LoadJson('./data/test_data.json')
    questions = []

    node_resps = [get_node_resp(model, node, config) for node in nodes]
    question_resps = [get_q_resp(model, question, config) for question in questions]
    node_resps = torch.stack(node_resps).squeeze()
    node_resps_norm = torch.norm(node_resps, dim=1)
    question_resps = torch.stack(question_resps).squeeze()
    question_resps_norm = torch.norm(question_resps, dim=1)

    scores = torch.mm(question_resps, torch.transpose(node_resps, 0, 1)) / torch.mm(node_resps_norm, torch.transpose(question_resps_norm, 0, 1))
    for i, score in enumerate(scores):
        idx = score.argsort()[-10:][::-1]
        test_datas[i]['scores'] = score[idx]
        test_datas[i]['pred_nodes'] = nodes[idx]
    
    sampy.SaveJson(test_datas, './data/test_pred.json')