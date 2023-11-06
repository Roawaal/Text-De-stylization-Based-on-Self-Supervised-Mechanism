import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from collections import Counter
from torchtext.vocab import Vocab
from torch.utils.data import Dataset, DataLoader
import torch


class IMDBDataset(Dataset):
    def __init__(self, data, vocab, label_transform):
        self.data = data
        self.vocab = vocab
        self.label_transform = label_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        sentiment = self.data.iloc[idx, 1]

        # Convert text and label to integer indices
        tokenized_text = [self.vocab[token] for token in text.split()]
        label = self.label_transform[sentiment]
        label_seq = torch.tensor([label])
        return torch.tensor(tokenized_text), label_seq

    @staticmethod
    def collate_fn(batch, pad_id):
        texts, labels = zip(*batch)
        texts_padded = pad_sequence(texts, batch_first=True, padding_value=pad_id)
        masks = torch.where(texts_padded != pad_id, torch.tensor(1), torch.tensor(0))
        labels = torch.stack(labels)
        return texts_padded, masks, labels


class DataPreprocessor:
    def __init__(self, file_path, tokenizer=lambda x: x.split()):
        # Load entire dataset for vocab
        full_data = pd.read_csv(file_path)
        full_data.columns = ['text', 'sentiment']

        # Trim to first 300 rows for training and testing
        self.data = full_data.head(300)

        # Tokenize text
        self.tokenizer = tokenizer
        self.tokens = [self.tokenizer(text) for text in full_data['text']]

        # Build vocabulary on entire dataset
        counter = Counter([token for sublist in self.tokens for token in sublist])
        counter.update({'<sos>': 1, '<eos>': 1, '<pad>': 1})
        self.vocab = Vocab(counter)

        # Label transformation
        self.label_transform = {'negative': 0, 'positive': 2}

    def split_data(self, train_size=0.833, val_size=0, test_size=0.167):
        assert train_size + val_size + test_size == 1, "Data split ratios do not sum to 1!"

        train, test = train_test_split(self.data, test_size=1 - train_size, random_state=42,
                                       stratify=self.data['sentiment'])

        return train, test

    def create_datasets(self, train_data, test_data, batch_size=32):
        train_dataset = IMDBDataset(train_data, self.vocab, self.label_transform)
        test_dataset = IMDBDataset(test_data, self.vocab, self.label_transform)
        pad_id = self.vocab['<pad>']

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: IMDBDataset.collate_fn(batch, pad_id))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: IMDBDataset.collate_fn(batch, pad_id))

        return train_loader, test_loader

    def get_vocab(self):
        return self.vocab

    def get_labels_list(self):
        return self.data['sentiment'].unique().tolist()


