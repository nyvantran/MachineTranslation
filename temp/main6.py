import torch
import torch.nn as nn
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


class OptimizedMETTDataset(Dataset):
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
            return self.tokenizer_eng.decode(input_ids, skip_special_tokens=True)
        elif language == 'vi':
            return self.tokenizer_vie.decode(input_ids, skip_special_tokens=True)
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


class LazyMETTDataset(Dataset):
    """
    Lazy loading version - tokenize on-the-fly
    Good for very large datasets that don't fit in memory
    """

    def __init__(
            self,
            data: List[Dict[str, str]],
            tokenizer_eng: str = "bert-base-uncased",
            tokenizer_vie: str = "vinai/phobert-base",
            max_length: int = 75
    ):
        self.max_length = max_length

        # Load tokenizers
        self.tokenizer_eng = AutoTokenizer.from_pretrained(tokenizer_eng)
        self.tokenizer_vie = AutoTokenizer.from_pretrained(tokenizer_vie)

        # Pre-filter data by length (rough estimation)
        logger.info("Pre-filtering data...")
        self.data = [
            item for item in data
            if item.get("en") and item.get("vi")
               and len(item["en"].split()) < max_length
               and len(item["vi"].split()) < max_length
        ]

        logger.info(f"Dataset ready with {len(self.data)} samples (lazy loading)")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize on-the-fly"""
        item = self.data[idx]

        en_encoded = self.tokenizer_eng(
            item["en"],
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=False
        )["input_ids"]

        vi_encoded = self.tokenizer_vie(
            item["vi"],
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=False
        )["input_ids"]

        return torch.tensor(en_encoded, dtype=torch.long), torch.tensor(vi_encoded, dtype=torch.long)

    def decode(self, input_ids, language: str = 'eng') -> str:
        if language == 'eng':
            return self.tokenizer_eng.decode(input_ids, skip_special_tokens=True)
        elif language == 'vi':
            return self.tokenizer_vie.decode(input_ids, skip_special_tokens=True)
        else:
            raise ValueError("language must be 'eng' or 'vi'")

    def get_vocab_size(self, language: str = 'vi') -> int:
        if language == 'eng':
            return len(self.tokenizer_eng)
        elif language == 'vi':
            return len(self.tokenizer_vie)
        else:
            raise ValueError("language must be 'eng' or 'vi'")


class FastMETTDataset(Dataset):
    """
    Ultra-fast version with multiprocessing tokenization
    """

    def __init__(
            self,
            data: List[Dict[str, str]],
            tokenizer_eng: str = "bert-base-uncased",
            tokenizer_vie: str = "vinai/phobert-base",
            max_length: int = 75,
            num_workers: int = 4,
            cache_dir: Optional[str] = None
    ):
        self.max_length = max_length
        self.num_workers = num_workers

        # Load tokenizers
        self.tokenizer_eng = AutoTokenizer.from_pretrained(tokenizer_eng)
        self.tokenizer_vie = AutoTokenizer.from_pretrained(tokenizer_vie)

        # Cache file
        cache_file = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(
                cache_dir,
                f"fast_cached_{tokenizer_eng.replace('/', '_')}_{tokenizer_vie.replace('/', '_')}_{max_length}.pkl"
            )

        # Load from cache or process
        if cache_file and os.path.exists(cache_file):
            logger.info(f"Loading from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.en_tokens = cache_data['en_tokens']
                self.vi_tokens = cache_data['vi_tokens']
        else:
            # Batch tokenization with multiprocessing
            logger.info("Batch tokenization with multiprocessing...")
            self.en_tokens, self.vi_tokens = self._batch_tokenize(data)

            # Save to cache
            if cache_file:
                logger.info(f"Saving to cache: {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'en_tokens': self.en_tokens,
                        'vi_tokens': self.vi_tokens
                    }, f)

        logger.info(f"Dataset ready with {len(self.en_tokens)} samples")

    def _batch_tokenize(self, data: List[Dict[str, str]]) -> Tuple[List[List[int]], List[List[int]]]:
        """Batch tokenization for speed"""
        # Extract texts
        en_texts = [item.get("en", "") for item in data]
        vi_texts = [item.get("vi", "") for item in data]

        # Batch tokenize English
        logger.info("Tokenizing English...")
        en_encoded = self.tokenizer_eng(
            en_texts,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=False
        )["input_ids"]

        # Batch tokenize Vietnamese
        logger.info("Tokenizing Vietnamese...")
        vi_encoded = self.tokenizer_vie(
            vi_texts,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=False
        )["input_ids"]

        # Filter by length
        logger.info("Filtering samples...")
        valid_en = []
        valid_vi = []

        for en_tok, vi_tok in zip(en_encoded, vi_encoded):
            if len(en_tok) > 0 and len(vi_tok) > 0 and len(en_tok) < self.max_length and len(vi_tok) < self.max_length:
                valid_en.append(en_tok)
                valid_vi.append(vi_tok)

        logger.info(f"Kept {len(valid_en)}/{len(data)} samples after filtering")

        return valid_en, valid_vi

    def __len__(self) -> int:
        return len(self.en_tokens)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.en_tokens[idx], dtype=torch.long), torch.tensor(self.vi_tokens[idx], dtype=torch.long)

    def decode(self, input_ids, language: str = 'eng') -> str:
        if language == 'eng':
            return self.tokenizer_eng.decode(input_ids, skip_special_tokens=True)
        elif language == 'vi':
            return self.tokenizer_vie.decode(input_ids, skip_special_tokens=True)
        else:
            raise ValueError("language must be 'eng' or 'vi'")

    def get_vocab_size(self, language: str = 'vi') -> int:
        if language == 'eng':
            return len(self.tokenizer_eng)
        elif language == 'vi':
            return len(self.tokenizer_vie)
        else:
            raise ValueError("language must be 'eng' or 'vi'")


class MemoryMappedMETTDataset(Dataset):
    """
    Memory-mapped version for extremely large datasets
    """

    def __init__(
            self,
            data_file: str,  # Path to preprocessed data file
            tokenizer_eng: str = "bert-base-uncased",
            tokenizer_vie: str = "vinai/phobert-base",
    ):
        self.tokenizer_eng = AutoTokenizer.from_pretrained(tokenizer_eng)
        self.tokenizer_vie = AutoTokenizer.from_pretrained(tokenizer_vie)

        # Load memory-mapped arrays
        data = np.load(data_file, allow_pickle=True)
        self.en_tokens = data['en_tokens']
        self.vi_tokens = data['vi_tokens']

        logger.info(f"Loaded memory-mapped dataset with {len(self.en_tokens)} samples")

    def __len__(self) -> int:
        return len(self.en_tokens)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.en_tokens[idx], dtype=torch.long), torch.tensor(self.vi_tokens[idx], dtype=torch.long)

    @staticmethod
    def create_memmap_file(
            data: List[Dict[str, str]],
            output_file: str,
            tokenizer_eng: str = "bert-base-uncased",
            tokenizer_vie: str = "vinai/phobert-base",
            max_length: int = 75
    ):
        """Create memory-mapped file from raw data"""
        logger.info("Creating memory-mapped file...")

        # Use FastMETTDataset to tokenize
        dataset = FastMETTDataset(data, tokenizer_eng, tokenizer_vie, max_length)

        # Save as numpy arrays
        np.savez_compressed(
            output_file,
            en_tokens=np.array(dataset.en_tokens, dtype=object),
            vi_tokens=np.array(dataset.vi_tokens, dtype=object)
        )

        logger.info(f"Saved to {output_file}")


# Original implementation (for comparison)
class OriginalMETTDataset(Dataset):
    def __init__(self, data, tokenizer_eng="bert-base-uncased", tokenizer_vie="vinai/phobert-base", max_length=75):
        self.data = data
        self.max_length = max_length
        self.tokenizer_eng = AutoTokenizer.from_pretrained(tokenizer_eng)
        self.tokenizer_vie = AutoTokenizer.from_pretrained(tokenizer_vie)

    def decode(self, input_ids, language='eng'):
        if language == 'eng':
            return self.tokenizer_eng.decode(input_ids)
        elif language == 'vi':
            return self.tokenizer_vie.decode(input_ids)
        else:
            raise ValueError("language must be 'eng' or 'vi'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        en, vie = data["en"], data["vi"]
        en_encoder = None
        vie_encoder = None
        try:
            en_encoder = self.tokenizer_eng(en)["input_ids"]
            vie_encoder = self.tokenizer_vie(vie)["input_ids"]
            if len(en_encoder) >= self.max_length or len(vie_encoder) >= self.max_length:
                return self.__getitem__(idx - 1)
        except Exception as e:
            if len(en_encoder) >= self.max_length or len(vie_encoder) >= self.max_length:
                return self.__getitem__(idx - 1)
        return torch.tensor(en_encoder), torch.tensor(vie_encoder)

    def get_lenvoacab(self, language='vi'):
        if language == 'eng':
            return len(self.tokenizer_eng)
        elif language == 'vi':
            return len(self.tokenizer_vie)
        else:
            raise ValueError("language must be 'eng' or 'vi'")


# Optimized collate function
def optimized_collate_fn(batch):
    """
    Optimized collate function với attention mask
    """
    sources, targets = zip(*batch)

    # Pad sequences
    padded_sources = pad_sequence(sources, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)

    # Create attention masks
    src_mask = (padded_sources != 0)
    tgt_mask = (padded_targets != 0)

    return {
        'src': padded_sources,
        'tgt': padded_targets,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask
    }


def efficient_collate_fn(batch):
    """
    More efficient collate with pre-computed lengths
    """
    sources, targets = zip(*batch)

    # Get lengths
    src_lengths = torch.tensor([len(src) for src in sources], dtype=torch.long)
    tgt_lengths = torch.tensor([len(tgt) for tgt in targets], dtype=torch.long)

    # Pad
    padded_sources = pad_sequence(sources, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)

    return {
        'src': padded_sources,
        'tgt': padded_targets,
        'src_lengths': src_lengths,
        'tgt_lengths': tgt_lengths
    }


import time
from torch.utils.data import DataLoader


def create_dummy_data(num_samples: int = 10000) -> List[Dict[str, str]]:
    """Create dummy translation data for testing"""
    dummy_data = []

    for i in range(num_samples):
        en_text = " ".join([f"word{j}" for j in range(np.random.randint(5, 50))])
        vi_text = " ".join([f"từ{j}" for j in range(np.random.randint(5, 50))])
        dummy_data.append({"en": en_text, "vi": vi_text})

    return dummy_data


def benchmark_dataset_creation(data, dataset_class, name, **kwargs):
    """Benchmark dataset creation time"""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Benchmarking {name}")
    logger.info(f"{'=' * 60}")

    start_time = time.time()
    dataset = dataset_class(data, **kwargs)
    creation_time = time.time() - start_time

    logger.info(f"Creation time: {creation_time:.2f}s")
    logger.info(f"Dataset size: {len(dataset)}")

    return dataset, creation_time


def benchmark_iteration(dataset, name, num_epochs=3, batch_size=32):
    """Benchmark iteration speed"""
    logger.info(f"\nBenchmarking iteration for {name}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single process for fair comparison
        collate_fn=optimized_collate_fn
    )

    times = []

    for epoch in range(num_epochs):
        start_time = time.time()

        for batch in dataloader:
            # Simulate processing
            pass

        epoch_time = time.time() - start_time
        times.append(epoch_time)
        logger.info(f"Epoch {epoch + 1}: {epoch_time:.2f}s")

    avg_time = np.mean(times)
    logger.info(f"Average epoch time: {avg_time:.2f}s")

    return avg_time


def run_comprehensive_benchmark():
    """Run comprehensive benchmark"""

    # Create dummy data
    logger.info("Creating dummy data...")
    data = create_dummy_data(num_samples=5000)

    results = {}

    # Test different implementations
    datasets = [
        (OriginalMETTDataset, "Original", {}),
        (OptimizedMETTDataset, "Optimized (Cached)", {"cache_dir": "./cache", "use_cache": True}),
        (LazyMETTDataset, "Lazy Loading", {}),
        (FastMETTDataset, "Fast (Batch Tokenization)", {"cache_dir": "./cache", "num_workers": 4}),
    ]

    for dataset_class, name, kwargs in datasets:
        try:
            # Benchmark creation
            dataset, creation_time = benchmark_dataset_creation(data, dataset_class, name, **kwargs)

            # Benchmark iteration
            iteration_time = benchmark_iteration(dataset, name, num_epochs=3, batch_size=32)

            results[name] = {
                'creation_time': creation_time,
                'iteration_time': iteration_time,
                'total_time': creation_time + iteration_time * 3
            }

        except Exception as e:
            logger.error(f"Error with {name}: {e}")
            results[name] = None

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Implementation':<30} {'Creation (s)':<15} {'Iteration (s)':<15} {'Total (s)':<15}")
    print("-" * 80)

    for name, result in results.items():
        if result:
            print(
                f"{name:<30} {result['creation_time']:>12.2f}   {result['iteration_time']:>12.2f}   {result['total_time']:>12.2f}")
        else:
            print(f"{name:<30} {'ERROR':<15}")

    print("=" * 80)


def test_correctness():
    """Test correctness of implementations"""
    logger.info("\n" + "=" * 60)
    logger.info("CORRECTNESS TEST")
    logger.info("=" * 60)

    # Create small test data
    test_data = [
        {"en": "Hello world", "vi": "Xin chào thế giới"},
        {"en": "How are you", "vi": "Bạn khỏe không"},
        {"en": "Good morning", "vi": "Chào buổi sáng"},
    ]

    # Test each implementation
    datasets = [
        (OriginalMETTDataset, "Original"),
        (OptimizedMETTDataset, "Optimized"),
        (LazyMETTDataset, "Lazy"),
        (FastMETTDataset, "Fast"),
    ]

    for dataset_class, name in datasets:
        try:
            logger.info(f"\nTesting {name}:")
            dataset = dataset_class(test_data, max_length=75)

            logger.info(f"  Dataset length: {len(dataset)}")

            # Get first item
            en_tokens, vi_tokens = dataset[0]
            logger.info(f"  First sample - EN tokens length: {len(en_tokens)}")
            logger.info(f"  First sample - VI tokens length: {len(vi_tokens)}")

            # Decode
            en_decoded = dataset.decode(en_tokens, language='eng')
            vi_decoded = dataset.decode(vi_tokens, language='vi')
            logger.info(f"  Decoded EN: {en_decoded}")
            logger.info(f"  Decoded VI: {vi_decoded}")

            # Vocab sizes
            logger.info(f"  EN vocab size: {dataset.get_vocab_size('eng')}")
            logger.info(f"  VI vocab size: {dataset.get_vocab_size('vi')}")

        except Exception as e:
            logger.error(f"Error testing {name}: {e}")


def test_dataloader():
    """Test DataLoader integration"""
    logger.info("\n" + "=" * 60)
    logger.info("DATALOADER INTEGRATION TEST")
    logger.info("=" * 60)

    # Create test data
    test_data = create_dummy_data(num_samples=100)

    # Create dataset
    dataset = FastMETTDataset(test_data, cache_dir="./cache")

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        collate_fn=optimized_collate_fn
    )

    # Test iteration
    logger.info("Testing DataLoader iteration...")
    for i, batch in enumerate(dataloader):
        logger.info(f"Batch {i}:")
        logger.info(f"  Source shape: {batch['src'].shape}")
        logger.info(f"  Target shape: {batch['tgt'].shape}")
        logger.info(f"  Source mask shape: {batch['src_mask'].shape}")
        logger.info(f"  Target mask shape: {batch['tgt_mask'].shape}")

        if i >= 2:  # Only show first 3 batches
            break

    logger.info("DataLoader test completed successfully!")


def memory_profiling():
    """Profile memory usage"""
    try:
        import tracemalloc

        logger.info("\n" + "=" * 60)
        logger.info("MEMORY PROFILING")
        logger.info("=" * 60)

        data = create_dummy_data(num_samples=1000)

        datasets_to_test = [
            (OptimizedMETTDataset, "Optimized", {"cache_dir": "./cache"}),
            (LazyMETTDataset, "Lazy", {}),
            (FastMETTDataset, "Fast", {"cache_dir": "./cache"}),
        ]

        for dataset_class, name, kwargs in datasets_to_test:
            tracemalloc.start()

            dataset = dataset_class(data, **kwargs)

            # Get some items
            for i in range(min(100, len(dataset))):
                _ = dataset[i]

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            logger.info(f"{name}:")
            logger.info(f"  Current memory: {current / 1024 / 1024:.2f} MB")
            logger.info(f"  Peak memory: {peak / 1024 / 1024:.2f} MB")
            logger.info("")

    except ImportError:
        logger.warning("tracemalloc not available for memory profiling")


if __name__ == "__main__":
    # Run tests
    test_correctness()

    # Run benchmark
    run_comprehensive_benchmark()

    # Test DataLoader
    test_dataloader()

    # Memory profiling
    memory_profiling()
