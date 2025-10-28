import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


class METTDataset(Dataset):
    def __init__(self, data, tokenizer_eng="bert-base-uncased", tokenizer_vie="vinai/phobert-base"):
        """
        Args:
            data (list of tuples): Each tuple contains (source_sequence, target_sequence)
        """
        self.data = data
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
        try:
            en_encoder = self.tokenizer_eng(en)["input_ids"]
            vie_encoder = self.tokenizer_vie(vie)["input_ids"]
        except Exception as e:
            en_encoder = self.tokenizer_eng("")["input_ids"]
            vie_encoder = self.tokenizer_vie("")["input_ids"]
            raise e
        return torch.tensor(en_encoder), torch.tensor(vie_encoder)

    def get_lenvoacab(self, language='vi'):
        if language == 'eng':
            return len(self.tokenizer_eng)
        elif language == 'vi':
            return len(self.tokenizer_vie)
        else:
            raise ValueError("language must be 'eng' or 'vi'")


def collate_fn(data):
    sources, targets = zip(*data)
    source_lengths = [len(src) for src in sources]
    padded_sources = pad_sequence(sources, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return padded_sources, padded_targets, source_lengths


def main():
    from datasets import load_dataset
    dataset = load_dataset('hiimbach/mtet', cache_dir='../datasets')["train"]
    mtet_dataset = METTDataset(data=dataset)
    datatest = mtet_dataset[0]
    en, vie = None, None
    try:
        en_encoder = mtet_dataset.tokenizer_eng(en)["input_ids"]
        vie_encoder = mtet_dataset.tokenizer_vie(vie)["input_ids"]
    except Exception as e:
        en_encoder = mtet_dataset.tokenizer_eng("")["input_ids"]
        vie_encoder = mtet_dataset.tokenizer_vie("")["input_ids"]
        print(e)

    print("en_encoder:", en_encoder)
    print("vie_encoder:", vie_encoder)


if __name__ == "__main__":
    main()
