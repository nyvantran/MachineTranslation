import torch
from tqdm import tqdm

from torch.utils.data import DataLoader
from datasets import load_dataset

import utils.load_save_model as load_save_model
from core.config import *
from core.model import Transformer
from core.dataset import METTDataset, collate_fn


# from core.loss import CrossEntropyLoss

def main():
    pre_data = load_dataset('hiimbach/mtet', cache_dir='datasets')
    train_dataset = METTDataset(data=pre_data['train'])
    # test_dataset = METTDataset(data=pre_data['test'])
    model = Transformer(
        input_dim=train_dataset.get_lenvoacab(language='eng'),
        output_dim=train_dataset.get_lenvoacab(language='vi'),
        emb_dim=512,
    )
    data0 = train_dataset[0]
    print("text input:", train_dataset.decode(data0[0].tolist(), language='eng'))
    text = model.transalate(data0[0], max_len=50, start_symbol=0, end_symbol=2)
    print("Translated text:", train_dataset.decode(text.tolist(), language='vi'))


if __name__ == "__main__":
    main()
