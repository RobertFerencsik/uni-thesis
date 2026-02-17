import pandas as pd
import torch
from .tokenizer import SentencePieceTokenizer

class SpamHamDataset:

    def __init__(
        self,
        csv_path,
        tokenizer,
        max_length = 512,
        text_column = 'Message',
        label_column = 'Spam/Ham',
        ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        dataframe = pd.read_csv(csv_path)

        self.texts = dataframe[text_column].astype(str).tolist()
        self.labels = dataframe[label_column].astype(str).tolist()

        self.labels = [1 if label == 'spam' else 0 for label in self.labels]

        self.encoded_texts, self.lengths = self.tokenizer.encode_batch(self.texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        token_ids = torch.tensor(self.encoded_texts[index], dtype=torch.long)
        seq_length = self.lengths[index]

        attention_mask = torch.zeros_like(token_ids)
        attention_mask[:seq_length] = 1

        label = torch.tensor(self.labels[index], dtype=torch.float32)

        return token_ids, attention_mask, label