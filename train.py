# ------------------------------------------------------------
# Wolf AI — Model Architecture
# ------------------------------------------------------------
# This file defines the Transformer model used by Wolf AI.
# It matches the architecture described in config.json.
# ------------------------------------------------------------

import torch
import torch.nn as nn


class WolfAIModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        vocab_size = config["architecture"]["vocab_size"]
        hidden = config["architecture"]["hidden_size"]
        n_layers = config["architecture"]["num_hidden_layers"]
        n_heads = config["architecture"]["num_attention_heads"]
        ff_size = config["architecture"]["intermediate_size"]
        dropout = config["architecture"]["dropout"]

        # ------------------------------------------------------------
        # Embedding layer
        # ------------------------------------------------------------
        self.embedding = nn.Embedding(vocab_size, hidden)

        # ------------------------------------------------------------
        # Transformer layers
        # ------------------------------------------------------------
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=n_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation="gelu"
            )
            for _ in range(n_layers)
        ])

        # Stack the layers into a full encoder
        self.transformer = nn.TransformerEncoder(self.layers[0], n_layers)

        # ------------------------------------------------------------
        # Output projection layer
        # ------------------------------------------------------------
        self.output = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids):
        # input_ids shape: (batch, seq_len)
        x = self.embedding(input_ids)          # (batch, seq_len, hidden)
        x = x.transpose(0, 1)                  # Transformer expects (seq_len, batch, hidden)

        x = self.transformer(x)                # Run through transformer
        x = x.transpose(0, 1)                  # Back to (batch, seq_len, hidden)

        logits = self.output(x)                # Predict next token
        return logits
