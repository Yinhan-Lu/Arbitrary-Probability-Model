"""
Wikipedia dataset loader for GPT-2 training
Uses Hugging Face datasets and GPT-2 tokenizer
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
from typing import Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WikipediaDataset(Dataset):
    """
    Wikipedia dataset for language modeling

    Processes Wikipedia articles into tokenized sequences for GPT-2 training
    """

    def __init__(
        self,
        tokenizer,
        max_length=1024,
        split="train",
        streaming=False,
        num_samples=None,
        dataset_name="wikipedia",
        dataset_config="20220301.en"
    ):
        """
        Args:
            tokenizer: GPT-2 tokenizer instance
            max_length: Maximum sequence length
            split: Dataset split ("train" or "validation")
            streaming: Whether to use streaming mode (for large datasets)
            num_samples: Number of samples to use (None for all)
            dataset_name: Hugging Face dataset name
            dataset_config: Dataset configuration/version
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.streaming = streaming

        logger.info(f"Loading Wikipedia dataset: {dataset_name} ({dataset_config}), split={split}")

        # Load Wikipedia dataset from Hugging Face
        try:
            if streaming:
                self.dataset = load_dataset(
                    dataset_name,
                    dataset_config,
                    split=split,
                    streaming=True
                )
                if num_samples:
                    self.dataset = self.dataset.take(num_samples)
                    self.num_samples = num_samples
                else:
                    self.num_samples = None  # Unknown for streaming
            else:
                self.dataset = load_dataset(
                    dataset_name,
                    dataset_config,
                    split=split
                )
                if num_samples:
                    self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
                self.num_samples = len(self.dataset)

            logger.info(f"Dataset loaded successfully. Samples: {self.num_samples if self.num_samples else 'streaming'}")

        except Exception as e:
            logger.error(f"Failed to load Wikipedia dataset: {e}")
            logger.info("Falling back to a smaller dataset for testing...")

            # Fallback to a smaller dataset for testing
            self.dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            if num_samples:
                self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
            self.num_samples = len(self.dataset)
            logger.info(f"Using WikiText-2 dataset instead. Samples: {self.num_samples}")

    def __len__(self):
        if self.streaming or self.num_samples is None:
            # For streaming datasets, return a large number
            return 1_000_000
        return self.num_samples

    def __getitem__(self, idx):
        """
        Get a single tokenized sample

        Returns:
            Dict with 'input_ids' and 'attention_mask'
        """
        # Get text from dataset
        if self.streaming:
            # For streaming, we can't index directly
            # This is handled differently in the DataLoader
            item = next(iter(self.dataset.skip(idx).take(1)))
        else:
            item = self.dataset[idx]

        # Extract text (field name depends on dataset)
        if "text" in item:
            text = item["text"]
        elif "title" in item and "content" in item:
            # Wikipedia format: combine title and content
            text = item["title"] + "\n\n" + item["content"]
        else:
            text = str(item)

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }


class StreamingWikipediaDataset:
    """
    Streaming version of Wikipedia dataset for memory efficiency
    Better for large-scale training
    """

    def __init__(
        self,
        tokenizer,
        max_length=1024,
        split="train",
        buffer_size=10000,
        dataset_name="wikipedia",
        dataset_config="20220301.en"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer_size = buffer_size

        logger.info(f"Loading streaming Wikipedia dataset: {dataset_name}")

        try:
            self.dataset = load_dataset(
                dataset_name,
                dataset_config,
                split=split,
                streaming=True
            )
        except Exception as e:
            logger.error(f"Failed to load Wikipedia dataset: {e}")
            logger.info("Falling back to WikiText-2...")
            self.dataset = load_dataset(
                "wikitext",
                "wikitext-2-raw-v1",
                split=split,
                streaming=True
            )

    def __iter__(self):
        """Iterate over the dataset"""
        for item in self.dataset:
            # Extract text
            if "text" in item:
                text = item["text"]
            elif "title" in item and "content" in item:
                text = item["title"] + "\n\n" + item["content"]
            else:
                text = str(item)

            # Skip empty texts
            if not text.strip():
                continue

            # Tokenize
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            yield {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0)
            }


def get_dataloader(
    config,
    split="train",
    batch_size=8,
    num_workers=4,
    streaming=False,
    num_samples=None
):
    """
    Create a DataLoader for Wikipedia dataset

    Args:
        config: Model configuration (to get max_seq_len)
        split: Dataset split ("train" or "validation")
        batch_size: Batch size
        num_workers: Number of data loading workers
        streaming: Whether to use streaming mode
        num_samples: Number of samples to use (None for all)

    Returns:
        DataLoader instance
    """
    # Initialize tokenizer
    logger.info("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # GPT-2 tokenizer doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if streaming:
        # Use streaming dataset
        dataset = StreamingWikipediaDataset(
            tokenizer=tokenizer,
            max_length=config.max_seq_len,
            split=split
        )

        # For streaming, we use a different approach
        def streaming_collate_fn(batch):
            return {
                "input_ids": torch.stack([item["input_ids"] for item in batch]),
                "attention_mask": torch.stack([item["attention_mask"] for item in batch])
            }

        # Note: streaming dataset returns an iterable, not a Dataset
        # We need to handle it differently
        return dataset  # Return the iterable directly

    else:
        # Use regular dataset
        dataset = WikipediaDataset(
            tokenizer=tokenizer,
            max_length=config.max_seq_len,
            split=split,
            streaming=False,
            num_samples=num_samples
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True
        )

        return dataloader


if __name__ == "__main__":
    # Test the dataset
    from model.config import get_config

    print("Testing Wikipedia Dataset...")

    config = get_config("tiny")
    print(f"\nUsing config: max_seq_len={config.max_seq_len}")

    # Test with a small number of samples
    print("\nCreating DataLoader with 10 samples...")
    dataloader = get_dataloader(
        config=config,
        split="train",
        batch_size=2,
        num_workers=0,
        streaming=False,
        num_samples=10
    )

    print(f"DataLoader created. Number of batches: {len(dataloader)}")

    # Get one batch
    print("\nFetching first batch...")
    batch = next(iter(dataloader))

    print(f"Batch keys: {batch.keys()}")
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"attention_mask shape: {batch['attention_mask'].shape}")

    print("\nSample input_ids (first 50 tokens):")
    print(batch["input_ids"][0, :50])

    print("\nDataset test passed successfully!")
