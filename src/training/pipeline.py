from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data.dataset import SpamHamDataset
from src.data.tokenizer import SentencePieceTokenizer
from src.infrastructure.artifact_manager import ArtifactManager
from src.infrastructure.paths import PATHS, Paths
from src.models.bilstm import BiLSTMSpamClassifier
from src.training.trainer import Trainer


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
        num_epochs: int,
        *,
        paths: Optional[Paths] = None,
        artifact_manager: Optional[ArtifactManager] = None,
        device: Optional[torch.device] = None,
    ):
        self.paths = paths if paths is not None else PATHS
        self._artifacts = artifact_manager if artifact_manager is not None else ArtifactManager()
        self.device = device if device is not None else torch.device("cuda")

        self.tokenizer_path = Path(self.paths.lstm_tokenizer)
        self.train_csv = Path(self.paths.train_processed)
        self.val_csv = Path(self.paths.validation_processed)
        self.save_dir = Path(self.paths.models)

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
            max_length=self.max_length,
        )
        vocab_size = self.tokenizer.get_vocab_size()

        train_df = self._artifacts.load_csv(self.train_csv)
        val_df = self._artifacts.load_csv(self.val_csv)

        self.train_dataset = SpamHamDataset(
            train_df,
            self.tokenizer,
            max_length=self.max_length,
        )
        self.val_dataset = SpamHamDataset(
            val_df,
            self.tokenizer,
            max_length=self.max_length,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.model = BiLSTMSpamClassifier(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            dense_hidden=self.dense_hidden,
            padding_idx=self.tokenizer.pad_id,
        )
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()

        self.trainer = Trainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=self.device,
            max_grad_norm=self.max_grad_norm,
            save_dir=str(self.save_dir),
            artifact_manager=self._artifacts,
        )

    def train(self) -> Dict[str, Any]:
        self.training_history = self.trainer.train(num_epochs=self.num_epochs)
        return self.training_history

    def run(self) -> Dict[str, Any]:
        self.setup()
        return self.train()
