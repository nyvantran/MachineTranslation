import torch
from tqdm import tqdm

from torch.utils.data import DataLoader
from datasets import load_dataset

import utils.load_save_model as load_save_model
from core.config import *
from core.model import Transformer
from core.dataset import METTDataset, collate_fn
from core.loss import CrossEntropyLoss


def train_model(model, dataloader, optimizer, loss_fn, epoch=0):
    model.train()
    toal_loss = 0.0
    loop = tqdm(dataloader, total=len(dataloader), desc="Training")
    for idx, (x, y, length) in enumerate(loop):
        x, y = x.to(DEVICES), y.to(DEVICES)
        optimizer.zero_grad()
        output = model(x, y)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        toal_loss += loss.item()

        if idx % 100 == 0:
            loop.set_postfix(loss=loss.item(), epoch=epoch + 1, idx=idx)
    avg_loss = toal_loss / len(dataloader)
    return avg_loss


def main():
    pre_data = load_dataset('hiimbach/mtet', cache_dir='datasets')
    train_dataset = METTDataset(data=pre_data['train'])
    # test_dataset = METTDataset(data=pre_data['test'])
    model = Transformer(
        input_dim=train_dataset.get_lenvoacab(language='eng'),
        output_dim=train_dataset.get_lenvoacab(language='vi'),
        emb_dim=512,
    ).to(DEVICES)
    print(DEVICES)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    print("Start training...")

    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     drop_last=False
    # )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = CrossEntropyLoss(smoothing=SMOOTHING)
    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, loss_fn, epoch)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}")
        if (epoch + 1) % 5 == 0:
            load_save_model.save_model(model, optimizer, epoch, train_loss, {},
                                       f'transformer_epoch{epoch + 1}.pt')


if __name__ == '__main__':
    main()
