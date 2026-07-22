# ------------------------------------------------------------
# Wolf AI — Metrics Utility
# ------------------------------------------------------------
# Provides simple evaluation metrics for training:
# - cross entropy loss (already in train.py)
# - token accuracy
# - perplexity
# ------------------------------------------------------------

import math
import torch


# ------------------------------------------------------------
# 1. Token Accuracy
# ------------------------------------------------------------
def token_accuracy(logits, target_ids):
    """
    Computes how often the model predicts the correct next token.
    logits: (batch, seq_len, vocab_size)
    target_ids: (batch, seq_len)
    """

    # Get predicted token IDs
    predictions = torch.argmax(logits, dim=-1)

    # Compare predictions to targets
    correct = (predictions == target_ids).float()

    # Compute accuracy
    accuracy = correct.mean().item()
    return accuracy


# ------------------------------------------------------------
# 2. Perplexity
# ------------------------------------------------------------
def perplexity(loss_value):
    """
    Converts cross entropy loss into perplexity.
    Lower perplexity = better model.
    """
    try:
        return math.exp(loss_value)
    except OverflowError:
        return float("inf")


# ------------------------------------------------------------
# 3. Metric Summary (optional helper)
# ------------------------------------------------------------
def summarize_metrics(loss_value, accuracy_value):
    """
    Returns a clean dictionary of metrics.
    """
    return {
        "loss": round(loss_value, 4),
        "accuracy": round(accuracy_value, 4),
        "perplexity": round(perplexity(loss_value), 4)
    }
