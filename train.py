import gc

import torch
import logging
import warnings

from nltk.corpus.reader import mte

import core.config as config

from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from core.model import Transformer
from core.dataset import METTDataset, collate_fn
from core.loss import CrossEntropyLoss

# configure logging
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.utils.checkpoint:*")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train(ct_model, loss, train_loader, optimizer, device, epoch=None):
    ct_model.train()
    total_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch}" if epoch is not None else "Training")
    for idx, batch in enumerate(loop):
        en_input_ids = batch['en_input_ids'].to(device)
        vi_input_ids = batch['vi_input_ids'].to(device)

        optimizer.zero_grad()

        outputs = ct_model(
            src=en_input_ids,
            tgt=vi_input_ids
        )

        loss_value = loss(
            predict=outputs,
            target=vi_input_ids
        )

        loss_value.backward()
        optimizer.step()

        total_loss += loss_value.item()
        if (idx + 1) % 10 == 0:
            loop.set_postfix(epoch=epoch, loss=total_loss / (idx + 1), idx=idx)

        if idx % 25 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def main():
    dataset = load_dataset('hiimbach/mtet', cache_dir="/datasets")["train"]
    mtet_dataset = METTDataset(dataset, cache_dir="./cache", max_length=config.MAX_LEN, use_cache=True)
    pad_idx = (mtet_dataset.tokenizer_eng.pad_token_id, mtet_dataset.tokenizer_vie.pad_token_id)
    train_loader = DataLoader(
        mtet_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        # num_workers=config.NUM_WORKERS,
        collate_fn=lambda x: collate_fn(x, pad_idx_eng=pad_idx[0], pad_idx_vie=pad_idx[1])
    )
    print("device:", config.DEVICES)
    model = Transformer(
        src_vocab_size=mtet_dataset.get_vocab_size(language='eng'),
        tgt_vocab_size=mtet_dataset.get_vocab_size(language='vi'),
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_seq_len=config.MAX_LEN,
        dropout=0.1,
        use_gradient_checkpointing=True,
        pad_idx=pad_idx
    )
    model.to(config.DEVICES)
    criterion = CrossEntropyLoss(
        vocab_size=mtet_dataset.get_vocab_size(language='vi'),
        label_smoothing=config.SMOOTHING,
        pad_idx=pad_idx[1],
    )
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    for epoch in range(1, config.EPOCHS + 1):
        avg_loss = train(model, criterion, train_loader, optimizer, config.DEVICES, epoch)
        logger.info(f"Epoch [{epoch}/{config.EPOCHS}], Loss: {avg_loss:.4f}")
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"transformer_epoch_{epoch}.pth")

        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
