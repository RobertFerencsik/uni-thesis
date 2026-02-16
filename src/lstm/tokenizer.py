import sentencepiece as spm
from pathlib import Path

class SentencePieceTokenizer:
    def __init__(self, model_path, max_length: int = 512):
        self.model_path = Path(model_path)
        self.max_length = max_length
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(self.model_path))

        self.pad_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else 0
        self.unk_id = self.sp.unk_id() if self.sp.unk_id() >= 0 else 1
        self.bos_id = self.sp.bos_id() if self.sp.bos_id() >= 0 else 2
        self.eos_id = self.sp.eos_id() if self.sp.eos_id() >= 0 else 3

        self.vocab_size = self.sp.get_piece_size()
        

    def encode(self, text):
        token_ids = self.sp.encode(text, out_type=int)
        return token_ids

    def decode(self, token_ids):
        return self.sp.decode(token_ids)

    def encode_batch(
        self,
        texts
        ):
        encoded = [self.encode(text) for text in texts]
        encoded = [ids[:self.max_length] for ids in encoded]

        lengths = [len(ids) for  ids in encoded]
        max_length = max(lengths)

        padded = []
        for ids in encoded:
            pad_length =  max_length - len(ids)
            padded.append(ids + [self.pad_id] * pad_length)
            return padded, lengths

    def get_vocab_size(self):
        return self.vocab_size