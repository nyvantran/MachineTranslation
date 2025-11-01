import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


class METTDataset(Dataset):
    def __init__(self, data, tokenizer_eng="bert-base-uncased", tokenizer_vie="vinai/phobert-base", max_length=75):
        """
        Args:
            data (list of tuples): Each tuple contains (source_sequence, target_sequence)
        """
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
    en, vie = datatest[0], datatest[1]


if __name__ == "__main__":
    main()
