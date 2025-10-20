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
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset_with_retry(dataset_name, dataset_config, split, max_retries=3, retry_delay=5, streaming=False):
    """
    Load HuggingFace dataset with retry logic to handle temporary server errors

    Args:
        dataset_name: Name of the dataset
        dataset_config: Dataset configuration
        split: Dataset split
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        streaming: Whether to use streaming mode

    Returns:
        Loaded dataset or None if all attempts fail
    """
    import os

    # Try with mirror endpoint first if default fails
    endpoints = [None, 'https://hf-mirror.com']

    for endpoint in endpoints:
        if endpoint:
            logger.info(f"Trying with mirror endpoint: {endpoint}")
            os.environ['HF_ENDPOINT'] = endpoint
        else:
            # Clear any previous endpoint setting
            os.environ.pop('HF_ENDPOINT', None)

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to load dataset {dataset_name} (attempt {attempt + 1}/{max_retries})...")

                if streaming:
                    dataset = load_dataset(
                        dataset_name,
                        dataset_config,
                        split=split,
                        streaming=True
                    )
                else:
                    dataset = load_dataset(
                        dataset_name,
                        dataset_config,
                        split=split
                    )

                logger.info(f"Successfully loaded {dataset_name}!")
                return dataset

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")

                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"All {max_retries} attempts failed for {dataset_name}")

        # If all retries with this endpoint failed, try next endpoint
        if endpoint is None:
            logger.warning("Default endpoint failed, trying mirror...")

    return None


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

        # Try to load dataset with retry logic
        self.dataset = load_dataset_with_retry(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            max_retries=3,
            retry_delay=5,
            streaming=streaming
        )

        # If primary dataset failed, try fallback options
        if self.dataset is None:
            logger.warning("Primary dataset failed after retries. Trying fallback options...")

            # Fallback 1: Try WikiText-103 (larger, more robust)
            logger.info("Attempting fallback: WikiText-103...")
            self.dataset = load_dataset_with_retry(
                dataset_name="wikitext",
                dataset_config="wikitext-103-raw-v1",
                split=split,
                max_retries=2,
                retry_delay=3,
                streaming=streaming
            )

        # Fallback 2: Try WikiText-2 (smallest, most reliable)
        if self.dataset is None:
            logger.warning("WikiText-103 also failed. Trying WikiText-2...")
            self.dataset = load_dataset_with_retry(
                dataset_name="wikitext",
                dataset_config="wikitext-2-raw-v1",
                split=split,
                max_retries=2,
                retry_delay=3,
                streaming=streaming
            )

        # If all attempts failed, raise error
        if self.dataset is None:
            raise RuntimeError(
                "Failed to load any dataset. Please check:\n"
                "1. Internet connection\n"
                "2. HuggingFace Hub status (https://status.huggingface.co)\n"
                "3. Disk space for caching datasets\n"
                "Consider using cached datasets or downloading datasets manually."
            )

        # Process loaded dataset
        if streaming:
            if num_samples:
                self.dataset = self.dataset.take(num_samples)
                self.num_samples = num_samples
            else:
                self.num_samples = None  # Unknown for streaming
        else:
            if num_samples:
                self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
            self.num_samples = len(self.dataset)

        logger.info(f"Dataset loaded successfully. Samples: {self.num_samples if self.num_samples else 'streaming'}")

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

        # Try primary dataset with retry
        self.dataset = load_dataset_with_retry(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            max_retries=3,
            retry_delay=5,
            streaming=True
        )

        # Fallback to WikiText-103
        if self.dataset is None:
            logger.warning("Primary dataset failed. Trying WikiText-103...")
            self.dataset = load_dataset_with_retry(
                dataset_name="wikitext",
                dataset_config="wikitext-103-raw-v1",
                split=split,
                max_retries=2,
                retry_delay=3,
                streaming=True
            )

        # Fallback to WikiText-2
        if self.dataset is None:
            logger.warning("WikiText-103 failed. Trying WikiText-2...")
            self.dataset = load_dataset_with_retry(
                dataset_name="wikitext",
                dataset_config="wikitext-2-raw-v1",
                split=split,
                max_retries=2,
                retry_delay=3,
                streaming=True
            )

        # If all failed, raise error
        if self.dataset is None:
            raise RuntimeError(
                "Failed to load any streaming dataset. Please check your internet connection "
                "and HuggingFace Hub status."
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
