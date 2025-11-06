import torch
import gc

from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from torch.utils.data import DataLoader
from datasets import load_dataset

import utils.load_save_model as load_save_model
from core.config import *
from core.model import Transformer
from core.dataset import METTDataset, collate_fn
from core.loss import CrossEntropyLoss



def train_model(model, dataloader, optimizer, loss_fn, epoch, scaler):
    model.train()
    toal_loss = 0.0
    optimizer.zero_grad()
    loop = tqdm(dataloader, total=len(dataloader), desc="Training")
    for idx, (x, y, length) in enumerate(loop):
        x, y = x.to(DEVICES), y.to(DEVICES)
        with autocast(enabled=USE_AMP):
            outputs = model(x, y)

            loss = loss_fn(outputs, y)
            loss /= ACCUMULATION_STEPS

        if USE_AMP:
            scaler.scale(loss).backward()  # Scale loss trước khi backward
        else:
            loss.backward()

        if ((idx + 1) % ACCUMULATION_STEPS == 0) or (idx + 1 == len(dataloader)):
            if USE_AMP:
                scaler.unscale_(optimizer)  # Unscale gradients trước khi clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                optimizer.step()
            optimizer.zero_grad()
        toal_loss += loss.item() * ACCUMULATION_STEPS

        if (idx + 1) % 100 == 0:
            loop.set_postfix(epoch=epoch + 1, loss=toal_loss / (idx + 1), batch_idx=idx + 1)

        if idx % 25 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    return toal_loss / len(dataloader)


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
    # loop = tqdm(train_loader)
    # for idx, (x, y, length) in enumerate(loop):
    #     print(train_dataset.decode(x[0].tolist(), language='eng'))
    #     print(train_dataset.decode(y[0].tolist(), language='vi'))
    #     break
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     drop_last=False
    # )
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer_dict = {p: torch.optim.Adam([p], foreach=False) for p in model.parameters()}

    # def optimizer_hook(parameter) -> None:
    #     optimizer_dict[parameter].step()
    #     optimizer_dict[parameter].zero_grad()
    #
    # for p in model.parameters():
    #     p.register_post_accumulate_grad_hook(optimizer_hook)

    loss_fn = CrossEntropyLoss(smoothing=SMOOTHING)
    scaler = GradScaler(enabled=USE_AMP)
    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, loss_fn, epoch, scaler)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}")
        if (epoch + 1) % 5 == 0:
            load_save_model.save_model(model, epoch, train_loss, {},
                                       f'transformer_epoch{epoch + 1}.pt')
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    main()
