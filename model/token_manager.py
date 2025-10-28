"""
Token Manager for Arbitrary Conditional Probability Model

Manages special tokens needed for conditional modeling:
- [M]: Mask token for unknown values
- [BOS]: Beginning of sequence token (optional, can reuse EOS)
"""

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenManager:
    """
    Manages special tokens for arbitrary conditional probability modeling

    Special Tokens:
    - [M]: Mask token to replace unknown variables in input
    - [BOS]: Beginning of sequence (reuses EOS if not added separately)
    """

    def __init__(self, tokenizer=None, add_mask_token=True, add_bos_token=False):
        """
        Initialize token manager

        Args:
            tokenizer: GPT2Tokenizer instance (will create if None)
            add_mask_token: Whether to add [M] token
            add_bos_token: Whether to add separate [BOS] token (else reuses EOS)
        """
        if tokenizer is None:
            logger.info("Loading GPT-2 tokenizer...")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # GPT-2 tokenizer doesn't have pad token by default
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer
        self.original_vocab_size = len(tokenizer)

        # Track added tokens
        self.mask_token = None
        self.mask_token_id = None
        self.bos_token = None
        self.bos_token_id = None

        # Add special tokens
        if add_mask_token:
            self._add_mask_token()

        if add_bos_token:
            self._add_bos_token()
        else:
            # Reuse EOS as BOS
            self.bos_token = tokenizer.eos_token
            self.bos_token_id = tokenizer.eos_token_id

        logger.info(f"TokenManager initialized:")
        logger.info(f"  Original vocab size: {self.original_vocab_size}")
        logger.info(f"  New vocab size: {len(self.tokenizer)}")
        logger.info(f"  Mask token: {self.mask_token} (ID: {self.mask_token_id})")
        logger.info(f"  BOS token: {self.bos_token} (ID: {self.bos_token_id})")

    def _add_mask_token(self):
        """Add [M] mask token to vocabulary"""
        mask_token = "[M]"

        # Check if already exists
        if mask_token in self.tokenizer.get_vocab():
            self.mask_token_id = self.tokenizer.convert_tokens_to_ids(mask_token)
            logger.info(f"Mask token {mask_token} already exists with ID {self.mask_token_id}")
        else:
            # Add new token
            num_added = self.tokenizer.add_tokens([mask_token])
            self.mask_token_id = self.tokenizer.convert_tokens_to_ids(mask_token)
            logger.info(f"Added {num_added} new token(s): {mask_token} (ID: {self.mask_token_id})")

        self.mask_token = mask_token

    def _add_bos_token(self):
        """Add [BOS] beginning of sequence token"""
        bos_token = "[BOS]"

        # Check if already exists
        if bos_token in self.tokenizer.get_vocab():
            self.bos_token_id = self.tokenizer.convert_tokens_to_ids(bos_token)
            logger.info(f"BOS token {bos_token} already exists with ID {self.bos_token_id}")
        else:
            # Add new token
            num_added = self.tokenizer.add_tokens([bos_token])
            self.bos_token_id = self.tokenizer.convert_tokens_to_ids(bos_token)
            logger.info(f"Added {num_added} new token(s): {bos_token} (ID: {self.bos_token_id})")

        self.bos_token = bos_token

    def resize_model_embeddings(self, model):
        """
        Resize model embeddings to accommodate new tokens

        Args:
            model: GPT2Model instance

        Returns:
            Updated model with resized embeddings
        """
        old_vocab_size = model.config.vocab_size
        new_vocab_size = len(self.tokenizer)

        if old_vocab_size == new_vocab_size:
            logger.info("Vocab size unchanged, no resizing needed")
            return model

        logger.info(f"Resizing embeddings from {old_vocab_size} to {new_vocab_size}")

        # Get device from model
        device = model.wte.weight.device

        # Update config
        model.config.vocab_size = new_vocab_size

        # Resize token embeddings
        old_embeddings = model.wte.weight.data
        new_embeddings = nn.Embedding(new_vocab_size, model.config.n_embd).to(device)

        # Copy old weights
        new_embeddings.weight.data[:old_vocab_size] = old_embeddings

        # Initialize new token embeddings
        # Use small random values similar to original initialization
        for i in range(old_vocab_size, new_vocab_size):
            new_embeddings.weight.data[i].normal_(mean=0.0, std=0.02)

        # Replace embeddings
        model.wte = new_embeddings

        # Resize output layer (tied with embeddings)
        model.lm_head = nn.Linear(model.config.n_embd, new_vocab_size, bias=False).to(device)
        model.lm_head.weight = model.wte.weight  # Weight tying

        logger.info(f"Embeddings resized successfully")
        logger.info(f"  Token embedding shape: {model.wte.weight.shape}")
        logger.info(f"  LM head shape: {model.lm_head.weight.shape}")

        return model

    def get_special_token_ids(self):
        """
        Get dictionary of special token IDs

        Returns:
            Dict with token names and IDs
        """
        return {
            "mask_token_id": self.mask_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

    def get_tokenizer(self):
        """Get the tokenizer with special tokens"""
        return self.tokenizer

    def save_tokenizer(self, save_directory):
        """Save tokenizer to directory"""
        self.tokenizer.save_pretrained(save_directory)
        logger.info(f"Tokenizer saved to {save_directory}")

    @classmethod
    def load_tokenizer(cls, load_directory):
        """Load tokenizer from directory"""
        tokenizer = GPT2Tokenizer.from_pretrained(load_directory)
        logger.info(f"Tokenizer loaded from {load_directory}")
        return cls(tokenizer=tokenizer, add_mask_token=False, add_bos_token=False)


if __name__ == "__main__":
    # Test token manager
    print("=" * 80)
    print("Testing TokenManager")
    print("=" * 80)

    # Create token manager
    token_mgr = TokenManager(add_mask_token=True, add_bos_token=False)

    # Get special tokens
    special_tokens = token_mgr.get_special_token_ids()
    print("\nSpecial Token IDs:")
    for name, token_id in special_tokens.items():
        print(f"  {name}: {token_id}")

    # Test encoding with mask token
    tokenizer = token_mgr.get_tokenizer()
    text = "Hello [M] world [M]"
    encoded = tokenizer(text, return_tensors="pt")
    print(f"\nTest encoding: '{text}'")
    print(f"  input_ids: {encoded['input_ids']}")
    print(f"  decoded: '{tokenizer.decode(encoded['input_ids'][0])}'")

    # Test with model (dummy)
    print("\nTest model embedding resize:")
    from model.config import get_config
    from model.arbitrary_prob_gpt2 import GPT2Model

    config = get_config("nano")
    model = GPT2Model(config)

    print(f"  Before: vocab_size={model.config.vocab_size}, embedding shape={model.wte.weight.shape}")
    model = token_mgr.resize_model_embeddings(model)
    print(f"  After: vocab_size={model.config.vocab_size}, embedding shape={model.wte.weight.shape}")

    print("\n" + "=" * 80)
    print("✓ TokenManager test passed!")
    print("=" * 80)
