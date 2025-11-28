import core.config as config
from core.model import *
from core.dataset import *

from torch.utils.data import DataLoader
from datasets import load_dataset

# dataset = load_dataset('hiimbach/mtet', cache_dir="/datasets")["train"]
# mtet_dataset = METTDataset(dataset, cache_dir="./cache", max_length=config.MAX_LEN, use_cache=True)
# pad_idx = (mtet_dataset.tokenizer_eng.pad_token_id, mtet_dataset.tokenizer_vie.pad_token_id)
# train_loader = DataLoader(
#     mtet_dataset,
#     batch_size=config.BATCH_SIZE,
#     shuffle=True,
#     # num_workers=config.NUM_WORKERS,
#     collate_fn=lambda x: collate_fn(x, pad_idx_eng=pad_idx[0], pad_idx_vie=pad_idx[1])
# )
print("device:", config.DEVICES)
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    max_seq_len=config.MAX_LEN,
    dropout=0.1,
    use_gradient_checkpointing=True,
    pad_idx=0
)
# Access named parameters
for name, param in model.named_parameters():
    print(name)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total number of parameters: {total_params}')
