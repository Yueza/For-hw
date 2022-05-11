import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import logging
import os

import config
from utils import load_data, CLassifyDataset, train, accuracy
from model import Classify

if __name__ == '__main__':
    # 设置log
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=config.logging_file_name,
        filemode='a+',
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'run on {device}')

    train_text, train_label, dev_text, dev_label = load_data(config)

    train_dataset = CLassifyDataset(train_text, train_label, config)
    dev_dataset = CLassifyDataset(dev_text, dev_label, config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False)

    model = Classify(config).to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    total_steps = len(train_loader) * config.epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, total_steps//10, total_steps)

    #训练
    best_model_name = ""
    best_acc = 0.0
    for epoch in range(config.epoch):
        train_loss = train(train_loader, model, loss, optimizer, device, scheduler)
        logger.info(f"Epoch: [{epoch + 1} / {config.epoch}]  | Train Loss: {train_loss}")
        acc = accuracy(dev_loader, model, device)
        logger.info(f"Epoch: [{epoch + 1} / {config.epoch}] | Dev ACC: {acc}")
        if acc > best_acc:
            best_acc = acc
            if best_model_name != '':
                os.remove(best_model_name)
            best_model_name = os.path.join(config.save_path, f"{config.subj}_{config.batch_size}_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), best_model_name)
            logger.info(f"Model saved in {best_model_name} @Epoch {epoch + 1}")
    
    # 评估
    # best_model_name = os.path.join(config.save_path, 'nyt10_64_epoch_14.pth')
    model.load_state_dict(torch.load(best_model_name))
    acc = accuracy(dev_loader, model, device)
    print(f"Best Model '{best_model_name}' | Test ACC: {acc}")
    logger.info(f"Best Model '{best_model_name}' | Test ACC: {acc}")