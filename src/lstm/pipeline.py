from pathlib import Path
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader
import sys

from .tokenizer import SentencePieceTokenizer
from .dataset import SpamHamDataset
from .model import BiLSTMSpamClassifier
from .train import Trainer

src = Path(__file__).resolve().parent.parent
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from config.config import PATHS, PROJECT_ROOT


class LSTMTrainingPipeline:    
    def __init__(
        self,
        max_length: int,
        batch_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout_rate: float,
        dense_hidden: int,
        learning_rate: float,
        max_grad_norm: float,
        num_epochs: int
    ):
        self.tokenizer_path = Path(PATHS.lstm_tokenizer)
        self.train_csv = Path(PATHS.train_processed)
        self.val_csv = Path(PATHS.validation_processed)
        self.save_dir = Path(PATHS.models)
        
        self.max_length = max_length
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.dense_hidden = dense_hidden
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        
        # Components (initialized in setup)
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.trainer = None
        self.training_history = None
    
    def setup(self):
        self.tokenizer = SentencePieceTokenizer(
            str(self.tokenizer_path),
            max_length=self.max_length
        )
        vocab_size = self.tokenizer.get_vocab_size()
        self.train_dataset = SpamHamDataset(
            str(self.train_csv),
            self.tokenizer,
            max_length=self.max_length
        )
        self.val_dataset = SpamHamDataset(
            str(self.val_csv),
            self.tokenizer,
            max_length=self.max_length
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        self.model = BiLSTMSpamClassifier(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            dense_hidden=self.dense_hidden,
            padding_idx=self.tokenizer.pad_id
        )
        self.trainer = Trainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=None,
            learning_rate=self.learning_rate,
            max_grad_norm=self.max_grad_norm,
            save_dir=str(self.save_dir)
        )
    
    def train(self) -> Dict[str, Any]:
        self.training_history = self.trainer.train(
            num_epochs=self.num_epochs
        )        
        return self.training_history
    
    def run(self) -> Dict[str, Any]:
        self.setup()
        return self.train()
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'project_root': str(PROJECT_ROOT),
            'tokenizer_path': str(self.tokenizer_path),
            'train_csv': str(self.train_csv),
            'val_csv': str(self.val_csv),
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'dense_hidden': self.dense_hidden,
            'learning_rate': self.learning_rate,
            'max_grad_norm': self.max_grad_norm,
            'num_epochs': self.num_epochs,
            'save_dir': str(self.save_dir)
        }