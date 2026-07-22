# ------------------------------------------------------------
# Wolf AI — Preprocessing Utility
# ------------------------------------------------------------
# Cleans raw text before tokenization and training.
# Includes:
# - lowercasing
# - punctuation spacing
# - removing extra spaces
# - optional special token insertion
# ------------------------------------------------------------

import re


# ------------------------------------------------------------
# 1. Basic text cleaning
# ------------------------------------------------------------
def clean_text(text):
    """
    Basic preprocessing for Wolf AI.
    """

    # Lowercase
    text = text.lower()

    # Space out punctuation
    text = re.sub(r"([.,!?;:])", r" \1 ", text)

    # Remove weird characters
    text = re.sub(r"[^a-z0-9.,!?;:\s<>_/]", "", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ------------------------------------------------------------
# 2. Add Wolf AI special tokens
# ------------------------------------------------------------
def add_special_tokens(text, add_bos=True, add_eos=True):
    """
    Adds Wolf AI's special tokens around text.
    """

    if add_bos:
        text = "<wolf_start> " + text

    if add_eos:
        text = text + " <wolf_end>"

    return text


# ------------------------------------------------------------
# 3. Full preprocessing pipeline
# ------------------------------------------------------------
def preprocess(text, use_special_tokens=True):
    """
    Full preprocessing pipeline used before tokenization.
    """

    cleaned = clean_text(text)

    if use_special_tokens:
        cleaned = add_special_tokens(cleaned)

    return cleaned
