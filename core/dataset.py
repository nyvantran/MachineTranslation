import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import pickle
import os
from typing import List, Tuple, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class METTDataset(Dataset):
    """
    Optimized version với:
    - Pre-filtering data quá dài
    - Cached tokenization
    - Proper error handling
    - Memory efficient
    """

    def __init__(
            self,
            data: List[Dict[str, str]],
            tokenizer_eng: str = "bert-base-uncased",
            tokenizer_vie: str = "vinai/phobert-base",
            max_length: int = 75,
            cache_dir: Optional[str] = None,
            use_cache: bool = True
    ):
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.use_cache = use_cache

        # Load tokenizers
        logger.info("Loading tokenizers...")
        self.tokenizer_eng = AutoTokenizer.from_pretrained(tokenizer_eng)
        self.tokenizer_vie = AutoTokenizer.from_pretrained(tokenizer_vie)

        # Cache file path
        cache_file = None
        if cache_dir and use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(
                cache_dir,
                f"cached_data_{tokenizer_eng.replace('/', '_')}_{tokenizer_vie.replace('/', '_')}_{max_length}.pkl"
            )

        # Try to load from cache
        if cache_file and os.path.exists(cache_file):
            logger.info(f"Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.data = cache_data['data']
                self.en_tokens = cache_data['en_tokens']
                self.vi_tokens = cache_data['vi_tokens']
            logger.info(f"Loaded {len(self.data)} samples from cache")
        else:
            # Process and filter data
            logger.info("Processing and filtering data...")
            self.data, self.en_tokens, self.vi_tokens = self._process_data(data)

            # Save to cache
            if cache_file:
                logger.info(f"Saving to cache: {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'data': self.data,
                        'en_tokens': self.en_tokens,
                        'vi_tokens': self.vi_tokens
                    }, f)

        logger.info(f"Dataset ready with {len(self.data)} samples")

    def _process_data(self, raw_data: List[Dict[str, str]]) -> Tuple[List[Dict], List[List[int]], List[List[int]]]:
        """Process and filter data, return valid samples only"""
        valid_data = []
        en_tokens_list = []
        vi_tokens_list = []

        filtered_count = 0
        error_count = 0

        for idx, item in enumerate(tqdm(raw_data, desc="Tokenizing")):
            try:
                en_text = item.get("en", "")
                vi_text = item.get("vi", "")

                # Skip empty
                if not en_text or not vi_text:
                    filtered_count += 1
                    continue

                # Tokenize
                en_encoded = self.tokenizer_eng(
                    en_text,
                    add_special_tokens=True,
                    truncation=False,
                    return_attention_mask=False
                )["input_ids"]

                vi_encoded = self.tokenizer_vie(
                    vi_text,
                    add_special_tokens=True,
                    truncation=False,
                    return_attention_mask=False
                )["input_ids"]

                # Filter by length
                if len(en_encoded) >= self.max_length or len(vi_encoded) >= self.max_length:
                    filtered_count += 1
                    continue

                # Keep valid samples
                valid_data.append(item)
                en_tokens_list.append(en_encoded)
                vi_tokens_list.append(vi_encoded)

            except Exception as e:
                error_count += 1
                if error_count <= 5:  # Log first 5 errors
                    logger.warning(f"Error processing item {idx}: {e}")
                continue

        logger.info(f"Filtered {filtered_count} samples (too long or empty)")
        logger.info(f"Errors: {error_count} samples")
        logger.info(f"Valid samples: {len(valid_data)}")

        return valid_data, en_tokens_list, vi_tokens_list

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return pre-tokenized data"""
        en_tokens = self.en_tokens[idx]
        vi_tokens = self.vi_tokens[idx]

        return torch.tensor(en_tokens, dtype=torch.long), torch.tensor(vi_tokens, dtype=torch.long)

    def decode(self, input_ids, language: str = 'eng') -> str:
        """Decode token ids back to text"""
        if language == 'eng':
            return self.tokenizer_eng.decode(input_ids, skip_special_tokens=False)
        elif language == 'vi':
            return self.tokenizer_vie.decode(input_ids, skip_special_tokens=False)
        else:
            raise ValueError("language must be 'eng' or 'vi'")

    def get_vocab_size(self, language: str = 'vi') -> int:
        """Get vocabulary size"""
        if language == 'eng':
            return len(self.tokenizer_eng)
        elif language == 'vi':
            return len(self.tokenizer_vie)
        else:
            raise ValueError("language must be 'eng' or 'vi'")


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_idx_eng: int, pad_idx_vie: int) -> Dict[
    str, torch.Tensor]:
    """Collate function to pad sequences in a batch"""
    en_batch, vi_batch = zip(*batch)

    en_padded = pad_sequence(en_batch, batch_first=True, padding_value=pad_idx_eng)
    vi_padded = pad_sequence(vi_batch, batch_first=True, padding_value=pad_idx_vie)

    return {
        'en_input_ids': en_padded,
        'vi_input_ids': vi_padded
    }


def main():
    from datasets import load_dataset
    import core.config as config
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    dataset = load_dataset('hiimbach/mtet', cache_dir='../datasets')["train"]
    train_dataset = METTDataset(data=dataset, cache_dir="../cache", max_length=config.MAX_LEN, use_cache=True)
    pad_idx = (train_dataset.tokenizer_eng.pad_token_id, train_dataset.tokenizer_vie.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, pad_idx_eng=pad_idx[0], pad_idx_vie=pad_idx[1])
    )
    loop = tqdm(train_loader, desc="Loading Batches")
    for idx, batch in enumerate(loop):
        en_input_ids = batch['en_input_ids']
        vi_input_ids = batch['vi_input_ids']
        print(f"Batch {idx}:")
        for i in range(en_input_ids.size(0)):
            print(f" Sample {i}:")
            print("eng_input_ids:", train_dataset.decode(en_input_ids[i], language='eng'))
            print("vi_input_ids:", train_dataset.decode(vi_input_ids[i], language='vi'))
        if idx == 0:  # Just check first 3 batches
            break


if __name__ == "__main__":
    main()
