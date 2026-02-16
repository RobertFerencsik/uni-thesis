import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMSpamClassifier(nn.Module):
    
    def __init__(
        self,
        vocab_size,
        embedding_dim = 128,
        hidden_size = 128,
        num_layers = 1,
        dropout_rate = 0.5,
        dense_hidden = 32,
        padding_idx = 0
    ):

        super(BiLSTMSpamClassifier, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.dense_hidden = dense_hidden
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
                vocab_size,
                embedding_dim,
                padding_idx=padding_idx
            )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        lstm_output_size = hidden_size * 2

        self.dropout = nn.Dropout(dropout_rate)

        self.dense1 = nn.Linear(lstm_output_size, dense_hidden)
        self.dense2 = nn.Linear(dense_hidden, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        token_ids,
        attention_mask
    ):
        embedded = self.embedding(token_ids)
        
        attention_mask = attention_mask.unsqueeze(-1)
        embedded = embedded * attention_mask

        _, (hidden, _) = self.lstm(embedded)

        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        lstm_final = torch.cat([forward_hidden, backward_hidden], dim=1)

        lstm_final = self.dropout(lstm_final)

        dense_out = F.relu(self.dense1(lstm_final))
        dense_out = self.dropout(dense_out)
        output = self.dense2(dense_out)

        output = self.sigmoid(output) #do not use BCELoss with sigmoid TODO: fix this

        return output

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'dense_hidden': self.dense_hidden,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }        