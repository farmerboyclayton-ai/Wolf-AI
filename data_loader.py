# ------------------------------------------------------------
# Wolf AI — Data Loader
# ------------------------------------------------------------
# Loads training text, tokenizes it, pads sequences, and
# returns batches for training.
# ------------------------------------------------------------

import torch


# ------------------------------------------------------------
# 1. Simple tokenizer (same as train.py & inference.py)
# ------------------------------------------------------------
class SimpleTokenizer:
    def __init__(self, vocab_file):
        with open(vocab_file, "r") as f:
            self.vocab = [line.strip() for line in f.readlines()]

        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.vocab)}

    def encode(self, text):
        tokens = text.split()
        return [self.token_to_id.get(tok, self.token_to_id["<unk>"]) for tok in tokens]

    def decode(self, ids):
        return " ".join([self.id_to_token.get(i, "<unk>") for i in ids])


# ------------------------------------------------------------
# 2. Load raw training text
# ------------------------------------------------------------
def load_training_text(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    return lines


# ------------------------------------------------------------
# 3. Encode lines into token IDs
# ------------------------------------------------------------
def encode_dataset(lines, tokenizer):
    encoded = [tokenizer.encode(line) for line in lines]
    return encoded


# ------------------------------------------------------------
# 4. Pad sequences to same length
# ------------------------------------------------------------
def pad_sequences(encoded_lines, pad_id):
    max_len = max(len(seq) for seq in encoded_lines)
    padded = [
        seq + [pad_id] * (max_len - len(seq))
        for seq in encoded_lines
    ]
    return padded


# ------------------------------------------------------------
# 5. Create batches
# ------------------------------------------------------------
def create_batches(padded_data, batch_size=4):
    batches = []
    for i in range(0, len(padded_data), batch_size):
        batch = padded_data[i:i + batch_size]
        batch_tensor = torch.tensor(batch, dtype=torch.long)
        batches.append(batch_tensor)
    return batches


# ------------------------------------------------------------
# 6. Full data loader pipeline
# ------------------------------------------------------------
def load_dataset(config):
    tokenizer = SimpleTokenizer(config["tokenizer"]["vocab_file"])
    pad_id = tokenizer.token_to_id["<pad>"]

    # Load raw lines
    lines = load_training_text(config["paths"]["train_corpus"])

    # Encode
    encoded = encode_dataset(lines, tokenizer)

    # Pad
    padded = pad_sequences(encoded, pad_id)

    # Batch
    batches = create_batches(
        padded,
        batch_size=config["training"]["batch_size"]
    )

    return batches, tokenizer
