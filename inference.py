# ------------------------------------------------------------
# Wolf AI — Inference Script
# ------------------------------------------------------------
# Loads:
# - config.json
# - tokenizer files
# - model architecture
# - model weights
# Then performs text generation.
# ------------------------------------------------------------

import json
import torch
import torch.nn as nn

# ------------------------------------------------------------
# 1. Load configuration
# ------------------------------------------------------------
def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)

config = load_config()

# ------------------------------------------------------------
# 2. Load tokenizer
# ------------------------------------------------------------
# Minimal tokenizer for early Wolf AI versions (v0.1–v0.3)
# Later you can replace this with a full BPE tokenizer.

class SimpleTokenizer:
    def __init__(self, vocab_file):
        with open(vocab_file, "r") as f:
            self.vocab = [line.strip() for line in f.readlines()]

        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.vocab)}

    def encode(self, text):
        # Very simple tokenizer (placeholder)
        tokens = text.split()
        return [self.token_to_id.get(tok, self.token_to_id.get("<unk>")) for tok in tokens]

    def decode(self, ids):
        return " ".join([self.id_to_token.get(i, "<unk>") for i in ids])


tokenizer = SimpleTokenizer(config["tokenizer"]["vocab_file"])

# ------------------------------------------------------------
# 3. Build model architecture
# ------------------------------------------------------------
# A minimal GPT-style block for early Wolf AI versions.
# You can expand this later as the model grows.

class WolfAIModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        vocab_size = cfg["architecture"]["vocab_size"]
        hidden = cfg["architecture"]["hidden_size"]
        n_layers = cfg["architecture"]["num_hidden_layers"]

        self.embedding = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=cfg["architecture"]["num_attention_heads"],
                dim_feedforward=cfg["architecture"]["intermediate_size"],
                dropout=cfg["architecture"]["dropout"],
                activation="gelu"
            )
            for _ in range(n_layers)
        ])
        self.transformer = nn.TransformerEncoder(self.layers[0], n_layers)
        self.output = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.output(x)
        return logits


model = WolfAIModel(config)

# ------------------------------------------------------------
# 4. Load model weights
# ------------------------------------------------------------
weights_path = config["paths"]["model_weights"]
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()

# ------------------------------------------------------------
# 5. Generation loop
# ------------------------------------------------------------
def generate(prompt, max_new_tokens=50):
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)

    for _ in range(max_new_tokens):
        logits = model(input_tensor)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        input_ids.append(next_token)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        # Stop if EOS token appears
        eos = config["tokenizer"]["special_tokens"]["eos_token"]
        if tokenizer.id_to_token.get(next_token) == eos:
            break

    return tokenizer.decode(input_ids)

# ------------------------------------------------------------
# 6. Run inference
# ------------------------------------------------------------
if __name__ == "__main__":
    print("🐺 Wolf AI — Ready for inference")
    user_input = input("You: ")
    response = generate(user_input)
    print("Wolf AI:", response)
