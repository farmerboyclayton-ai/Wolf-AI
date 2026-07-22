# ------------------------------------------------------------
# Wolf AI — Config Loader Utility
# ------------------------------------------------------------
# Loads config.json and provides helper functions for accessing
# configuration values safely.
# ------------------------------------------------------------

import json
import os


class ConfigLoader:
    def __init__(self, path="config.json"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            self.config = json.load(f)

    # ------------------------------------------------------------
    # Get full config
    # ------------------------------------------------------------
    def get_all(self):
        return self.config

    # ------------------------------------------------------------
    # Get architecture settings
    # ------------------------------------------------------------
    def architecture(self):
        return self.config.get("architecture", {})

    # ------------------------------------------------------------
    # Get tokenizer settings
    # ------------------------------------------------------------
    def tokenizer(self):
        return self.config.get("tokenizer", {})

    # ------------------------------------------------------------
    # Get file paths
    # ------------------------------------------------------------
    def paths(self):
        return self.config.get("paths", {})

    # ------------------------------------------------------------
    # Get training settings
    # ------------------------------------------------------------
    def training(self):
        return self.config.get("training", {})

    # ------------------------------------------------------------
    # Safe getter
    # ------------------------------------------------------------
    def get(self, key, default=None):
        return self.config.get(key, default)
