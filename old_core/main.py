import torch
from tqdm import tqdm
import torchaudio

from torch.utils.data import DataLoader
from datasets import load_dataset

import utils.load_save_model as load_save_model
from old_core.config import *
from old_core.model import Transformer
from old_core.dataset import METTDataset, collate_fn
from torchviz import make_dot


# from old_core.loss import CrossEntropyLoss

def main():
    pre_data = load_dataset('hiimbach/mtet', cache_dir='../datasets')
    train_dataset = METTDataset(data=pre_data['train'])
    # test_dataset = METTDataset(data=pre_data['temp'])
    model = Transformer(
        input_dim=train_dataset.get_lenvoacab(language='eng'),
        output_dim=train_dataset.get_lenvoacab(language='vi'),
        emb_dim=512,
    ).to("cuda")
    model.cuda()
    x = train_dataset[0][0].unsqueeze(0).to("cuda")  # (1, src_seq_len)
    y = train_dataset[0][1].unsqueeze(0).to("cuda")  # (1, tgt_seq_len)
    output = model(x, y)
    loss = output.sum()
    dot = make_dot(loss, params=dict(model.named_parameters()))
    dot.render("linear_cuda_graph.png", format="png")


if __name__ == "__main__":
    main()
