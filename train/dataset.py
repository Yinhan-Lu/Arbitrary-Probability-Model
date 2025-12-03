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
    Wikipedia dataset for language modeling with concatenate + chunk approach

    Concatenates all documents with EOS separators, then chunks into fixed-length
    sequences. This eliminates placeholder text and maximizes data utilization.
    Follows mainstream practice (HuggingFace, GPT-2, etc.)
    """

    def __init__(
        self,
        tokenizer,
        max_length=1024,
        split="train",
        streaming=False,
        num_samples=None,
        dataset_name="wikitext",
        dataset_config="wikitext-103-raw-v1",
        primary_dataset_only=True
    ):
        """
        Args:
            tokenizer: GPT-2 tokenizer instance
            max_length: Maximum sequence length (chunk size)
            split: Dataset split ("train" or "validation")
            streaming: Whether to use streaming mode (not supported with chunking)
            num_samples: Number of documents to process (None for all)
            dataset_name: Hugging Face dataset name
            dataset_config: Dataset configuration/version
            primary_dataset_only: If True, only use specified dataset without fallback
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
            streaming=False  # Force non-streaming for chunking preprocessing
        )

        # If primary dataset failed, try fallback options (only if allowed)
        if self.dataset is None and not primary_dataset_only:
            logger.warning("Primary dataset failed after retries. Trying fallback options...")

            # Fallback 1: Try WikiText-103 (larger, more robust)
            logger.info("Attempting fallback: WikiText-103...")
            self.dataset = load_dataset_with_retry(
                dataset_name="wikitext",
                dataset_config="wikitext-103-raw-v1",
                split=split,
                max_retries=2,
                retry_delay=3,
                streaming=False
            )

        # Fallback 2: Try WikiText-2 (smallest, most reliable)
        if self.dataset is None and not primary_dataset_only:
            logger.warning("WikiText-103 also failed. Trying WikiText-2...")
            self.dataset = load_dataset_with_retry(
                dataset_name="wikitext",
                dataset_config="wikitext-2-raw-v1",
                split=split,
                max_retries=2,
                retry_delay=3,
                streaming=False
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

        # Limit dataset size if requested
        if num_samples:
            total_docs = len(self.dataset)
            self.dataset = self.dataset.select(range(min(num_samples, total_docs)))
            logger.info(f"Using {len(self.dataset)} documents out of {total_docs}")

        logger.info(f"Dataset loaded successfully. Documents: {len(self.dataset)}")

        # Preprocess: concatenate + chunk
        logger.info("Preprocessing: concatenating documents and chunking...")
        self.chunks = self._create_chunks()
        self.num_samples = len(self.chunks)

        logger.info(f"Created {self.num_samples} chunks of length {self.max_length}")

    def _create_chunks(self):
        """
        Concatenate all documents with EOS separators and chunk into fixed-length sequences

        Process:
        1. Iterate through all documents
        2. Tokenize each document
        3. Concatenate with EOS token separator
        4. Chunk concatenated sequence into max_length pieces
        5. Return list of chunks (only complete chunks)

        Returns:
            List of token ID lists, each of length self.max_length
        """
        all_tokens = []
        eos_token_id = self.tokenizer.eos_token_id

        total_docs = len(self.dataset)
        logger.info(f"Processing {total_docs} documents...")

        # Process all documents
        for idx in range(total_docs):
            item = self.dataset[idx]

            # Extract text (field name depends on dataset)
            if "text" in item:
                text = item["text"]
            elif "title" in item and "content" in item:
                text = item["title"] + "\n\n" + item["content"]
            else:
                text = str(item)

            # Skip completely empty strings (they produce 0 tokens anyway)
            # Don't filter short texts - they have information (headers, etc.)
            if not text:
                continue

            # Tokenize without truncation
            tokens = self.tokenizer.encode(text, add_special_tokens=False)

            # Skip if tokenization produced nothing
            if not tokens:
                continue

            # Add tokens + EOS separator
            all_tokens.extend(tokens)
            all_tokens.append(eos_token_id)

            # Progress logging every 10k documents
            if (idx + 1) % 10000 == 0:
                logger.info(f"  Processed {idx + 1}/{total_docs} documents, "
                          f"accumulated {len(all_tokens):,} tokens")

        logger.info(f"Total tokens accumulated: {len(all_tokens):,}")

        # Chunk into fixed-length sequences
        chunks = []
        num_complete_chunks = len(all_tokens) // self.max_length

        for i in range(num_complete_chunks):
            start_idx = i * self.max_length
            end_idx = start_idx + self.max_length
            chunk = all_tokens[start_idx:end_idx]
            chunks.append(chunk)

        # Report statistics
        num_dropped_tokens = len(all_tokens) % self.max_length
        dropped_pct = (num_dropped_tokens / len(all_tokens)) * 100 if all_tokens else 0
        logger.info(f"Created {len(chunks)} complete chunks")
        logger.info(f"Dropped {num_dropped_tokens} tokens ({dropped_pct:.2f}%) from incomplete last chunk")

        return chunks

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Get a pre-chunked sequence

        Returns:
            Dict with 'input_ids' and 'attention_mask'
            - All tokens are valid (no padding in chunks)
            - attention_mask is all 1s
        """
        chunk = self.chunks[idx]

        # Convert to tensor
        input_ids = torch.tensor(chunk, dtype=torch.long)

        # All positions are valid (no padding in chunks)
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }


class StreamingWikipediaDataset:
    """
    Streaming version of Wikipedia dataset with on-the-fly chunking

    Uses buffer to concatenate documents with EOS separators, then yields
    fixed-length chunks. Memory-efficient for large-scale training.
    """

    def __init__(
        self,
        tokenizer,
        max_length=1024,
        split="train",
        buffer_size=10000,
        dataset_name="wikitext",
        dataset_config="wikitext-103-raw-v1",
        primary_dataset_only=True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.eos_token_id = tokenizer.eos_token_id

        logger.info(f"Loading streaming Wikipedia dataset: {dataset_name}")
        logger.info(f"Chunk size: {max_length}, Buffer size: {buffer_size}")

        # Try primary dataset with retry
        self.dataset = load_dataset_with_retry(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            max_retries=3,
            retry_delay=5,
            streaming=True
        )

        # Fallback to WikiText-103 (only if allowed)
        if self.dataset is None and not primary_dataset_only:
            logger.warning("Primary dataset failed. Trying WikiText-103...")
            self.dataset = load_dataset_with_retry(
                dataset_name="wikitext",
                dataset_config="wikitext-103-raw-v1",
                split=split,
                max_retries=2,
                retry_delay=3,
                streaming=True
            )

        # Fallback to WikiText-2 (only if allowed)
        if self.dataset is None and not primary_dataset_only:
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
        """
        Iterate over the dataset, yielding fixed-length chunks

        Uses a buffer to concatenate documents, then yields chunks on-the-fly.
        """
        buffer = []
        chunks_yielded = 0

        for item in self.dataset:
            # Extract text
            if "text" in item:
                text = item["text"]
            elif "title" in item and "content" in item:
                text = item["title"] + "\n\n" + item["content"]
            else:
                text = str(item)

            # Skip completely empty strings (produce 0 tokens anyway)
            if not text:
                continue

            # Tokenize without truncation
            tokens = self.tokenizer.encode(text, add_special_tokens=False)

            # Skip if tokenization produced nothing
            if not tokens:
                continue

            # Add tokens + EOS separator to buffer
            buffer.extend(tokens)
            buffer.append(self.eos_token_id)

            # Yield complete chunks from buffer
            while len(buffer) >= self.max_length:
                chunk = buffer[:self.max_length]
                buffer = buffer[self.max_length:]

                # Convert to tensor
                input_ids = torch.tensor(chunk, dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)

                chunks_yielded += 1
                if chunks_yielded % 1000 == 0:
                    logger.debug(f"Yielded {chunks_yielded} chunks, buffer size: {len(buffer)}")

                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }


class SimpleCollateFn:
    """
    Simple collate function that only does dynamic padding (no augmentation)

    Implemented as a class to be picklable for multiprocessing in DataLoader.
    Used for validation dataloaders where evaluation modes apply their own augmentation.
    """

    def __init__(self, pad_token_id=50256):
        """
        Args:
            pad_token_id: Token ID to use for padding (default: 50256 for GPT-2)
        """
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        """
        Collate function: dynamic padding only, no augmentation

        Args:
            batch: List of dicts with 'input_ids' and 'attention_mask'

        Returns:
            Batched dict with dynamically padded tensors:
            - input_ids: (batch_size, max_len)
            - attention_mask: (batch_size, max_len)
        """
        # Find max length in batch
        max_len = max(item['input_ids'].size(0) for item in batch)

        # Pad each sample
        batch_input_ids = []
        batch_attention_masks = []

        for item in batch:
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']
            current_len = input_ids.size(0)
            pad_len = max_len - current_len

            if pad_len > 0:
                # Pad input_ids
                padded_input = torch.cat([
                    input_ids,
                    torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
                ])

                # Pad attention_mask
                padded_mask = torch.cat([
                    attention_mask,
                    torch.zeros(pad_len, dtype=torch.long)
                ])
            else:
                padded_input = input_ids
                padded_mask = attention_mask

            batch_input_ids.append(padded_input)
            batch_attention_masks.append(padded_mask)

        # Stack into batch tensors
        return {
            'input_ids': torch.stack(batch_input_ids),
            'attention_mask': torch.stack(batch_attention_masks)
        }


def create_simple_collate_fn(pad_token_id=50256):
    """
    Create simple collate function that only does dynamic padding (no augmentation)

    Used for validation dataloaders where evaluation modes apply their own augmentation.

    Args:
        pad_token_id: Token ID to use for padding (default: 50256 for GPT-2)

    Returns:
        SimpleCollateFn instance (picklable for multiprocessing)
    """
    return SimpleCollateFn(pad_token_id)


class IndicesSamplingCollateFn:
    """
    Collate function that samples conditioning/evaluation/unseen indices in parallel workers

    This is a performance optimization: instead of sampling indices serially in the main
    training process (which causes GPU to wait for CPU), we sample them in parallel
    DataLoader worker processes. This achieves 2-4x speedup by maximizing GPU utilization.

    The model interface remains unchanged - it still receives:
        (input_ids, conditional_idx, evaluation_idx, unseen_idx)

    Implemented as a class to be picklable for multiprocessing in DataLoader.
    """

    def __init__(self, augmenter, pad_token_id=50256, use_attention_mask_for_valid=True):
        """
        Args:
            augmenter: ConditionalAugmenter instance with split_indices() method
            pad_token_id: Token ID to use for padding (default: 50256 for GPT-2)
            use_attention_mask_for_valid: If True (default, new behavior), use attention_mask
                to determine valid positions. If False (old behavior), use pad_token_id
                to exclude positions. The old behavior incorrectly excludes EOS tokens
                since GPT-2's pad_token_id == eos_token_id == 50256.
        """
        self.augmenter = augmenter
        self.pad_token_id = pad_token_id
        self.use_attention_mask_for_valid = use_attention_mask_for_valid

    def __call__(self, batch):
        """
        Collate function: sample indices in worker, then dynamic padding

        This happens in DataLoader worker processes (parallel), not main process (serial).
        Each worker processes multiple samples concurrently, dramatically reducing CPU time.

        Args:
            batch: List of dicts with 'input_ids' and 'attention_mask'

        Returns:
            Batched dict with dynamically padded tensors + pre-sampled indices:
            - input_ids: (batch_size, max_len)
            - attention_mask: (batch_size, max_len)
            - conditional_idx: List[List[int]] - conditioning indices for each sample
            - evaluation_idx: List[List[int]] - evaluation indices for each sample
            - unseen_idx: List[List[int]] - unseen indices for each sample
        """
        # Step 1: Sample indices for each sample (happens in worker process)
        batch_cond_idx = []
        batch_eval_idx = []
        batch_unseen_idx = []

        for item in batch:
            input_ids = item['input_ids']
            seq_len = input_ids.size(0)
            attention_mask = item['attention_mask']

            # Find valid (non-padding) positions
            if self.use_attention_mask_for_valid:
                # New behavior: use attention_mask (includes EOS tokens)
                valid_positions = [i for i in range(seq_len) if attention_mask[i] == 1]
            else:
                # Old behavior: use pad_token_id (excludes EOS tokens incorrectly)
                valid_positions = [
                    i for i in range(seq_len)
                    if input_ids[i] != self.pad_token_id
                ]

            # Sample indices using augmenter
            cond_idx, eval_idx, unseen_idx = self.augmenter.split_indices(
                seq_len=seq_len,
                valid_positions=valid_positions
            )

            batch_cond_idx.append(cond_idx)
            batch_eval_idx.append(eval_idx)
            batch_unseen_idx.append(unseen_idx)

        # Step 2: Dynamic padding (same as SimpleCollateFn)
        max_len = max(item['input_ids'].size(0) for item in batch)

        batch_input_ids = []
        batch_attention_masks = []

        for item in batch:
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']
            current_len = input_ids.size(0)
            pad_len = max_len - current_len

            if pad_len > 0:
                # Pad input_ids
                padded_input = torch.cat([
                    input_ids,
                    torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
                ])

                # Pad attention_mask
                padded_mask = torch.cat([
                    attention_mask,
                    torch.zeros(pad_len, dtype=torch.long)
                ])
            else:
                padded_input = input_ids
                padded_mask = attention_mask

            batch_input_ids.append(padded_input)
            batch_attention_masks.append(padded_mask)

        # Step 3: Return batch with pre-sampled indices
        return {
            'input_ids': torch.stack(batch_input_ids),
            'attention_mask': torch.stack(batch_attention_masks),
            'conditional_idx': batch_cond_idx,  # Ready to use!
            'evaluation_idx': batch_eval_idx,
            'unseen_idx': batch_unseen_idx
        }


def create_indices_sampling_collate_fn(augmenter, pad_token_id=50256, use_attention_mask_for_valid=True):
    """
    Create collate function that samples indices in parallel DataLoader workers

    This is a performance optimization for conditional training:
    - Moves CPU-bound indices sampling from main process to worker processes
    - Achieves 2-4x speedup by parallelizing CPU work
    - GPU no longer waits for CPU to finish sampling

    Args:
        augmenter: ConditionalAugmenter instance
        pad_token_id: Token ID to use for padding (default: 50256 for GPT-2)
        use_attention_mask_for_valid: If True (default), use attention_mask to determine
            valid positions. If False, use pad_token_id (old buggy behavior).

    Returns:
        IndicesSamplingCollateFn instance (picklable for multiprocessing)
    """
    return IndicesSamplingCollateFn(augmenter, pad_token_id, use_attention_mask_for_valid)


class AugmentCollateFn:
    """
    Collate function that does augmentation + dynamic padding

    Implemented as a class to be picklable for multiprocessing in DataLoader.
    This is the optimal approach: augment first, then dynamically pad to
    the longest augmented sequence in the batch (single padding pass).
    """

    def __init__(self, augmenter, device='cpu'):
        """
        Args:
            augmenter: ConditionalAugmenter instance
            device: Device for augmentation ('cpu' recommended for DataLoader workers)
        """
        self.augmenter = augmenter
        self.device = device

    def __call__(self, batch):
        """
        Collate function: augment each sample, then dynamic padding

        Args:
            batch: List of dicts with 'input_ids' and 'attention_mask'

        Returns:
            Batched dict with augmented and dynamically padded tensors:
            - input_ids: (batch_size, max_aug_len)
            - position_ids: (batch_size, max_aug_len)
            - labels: (batch_size, max_aug_len)
            - attention_mask: (batch_size, 1, max_aug_len, max_aug_len)
        """
        # Step 1: Augment each sample
        augmented_samples = []
        for item in batch:
            # Extract sequences (remove batch dim from tokenizer)
            input_ids = item['input_ids'].squeeze(0) if item['input_ids'].dim() > 1 else item['input_ids']
            attention_mask = item['attention_mask'].squeeze(0) if item['attention_mask'].dim() > 1 else item['attention_mask']

            # Find valid (non-padding) positions
            valid_positions = [i for i in range(len(input_ids)) if attention_mask[i] == 1]

            # Augment this sample
            aug_result = self.augmenter.augment_sequence(
                input_ids,
                device=self.device,
                valid_positions=valid_positions
            )
            augmented_samples.append(aug_result)

        # Step 2: Find max length in augmented batch
        max_len = max(sample['aug_input_ids'].size(0) for sample in augmented_samples)

        # Step 3: Dynamic padding to max_len
        batch_input_ids = []
        batch_position_ids = []
        batch_labels = []
        batch_attention_masks = []

        for sample in augmented_samples:
            current_len = sample['aug_input_ids'].size(0)
            pad_len = max_len - current_len

            if pad_len > 0:
                # Pad input_ids with pad token (50256 for GPT-2)
                padded_input = torch.cat([
                    sample['aug_input_ids'],
                    torch.full((pad_len,), 50256, dtype=torch.long, device=self.device)
                ])

                # Pad position_ids with 0
                padded_position = torch.cat([
                    sample['position_ids'],
                    torch.zeros(pad_len, dtype=torch.long, device=self.device)
                ])

                # Pad labels with -100 (ignored by loss function)
                padded_label = torch.cat([
                    sample['labels'],
                    torch.full((pad_len,), -100, dtype=torch.long, device=self.device)
                ])

                # Pad attention mask (2D matrix)
                # Original shape: (current_len, current_len)
                # Target shape: (max_len, max_len)
                padded_mask = torch.zeros(max_len, max_len, dtype=sample['attention_mask'].dtype, device=self.device)
                padded_mask[:current_len, :current_len] = sample['attention_mask']
                # Let padding positions attend to valid content (prevents softmax NaN)
                padded_mask[current_len:, :current_len] = 1
            else:
                # No padding needed
                padded_input = sample['aug_input_ids']
                padded_position = sample['position_ids']
                padded_label = sample['labels']
                padded_mask = sample['attention_mask']

            batch_input_ids.append(padded_input)
            batch_position_ids.append(padded_position)
            batch_labels.append(padded_label)
            batch_attention_masks.append(padded_mask)

        # Step 4: Stack into batch tensors
        return {
            'input_ids': torch.stack(batch_input_ids),
            'position_ids': torch.stack(batch_position_ids),
            'labels': torch.stack(batch_labels),
            'attention_mask': torch.stack(batch_attention_masks).unsqueeze(1),  # Add head dim: (B, 1, L, L)
        }


def create_augment_collate_fn(augmenter, device='cpu'):
    """
    Create collate function that does augmentation + dynamic padding

    This is the optimal approach: augment first, then dynamically pad to
    the longest augmented sequence in the batch (single padding pass).

    Args:
        augmenter: ConditionalAugmenter instance
        device: Device for augmentation ('cpu' recommended for DataLoader workers)

    Returns:
        AugmentCollateFn instance (picklable for multiprocessing)
    """
    return AugmentCollateFn(augmenter, device)


def worker_init_fn(worker_id):
    """
    Initialize random seed for each DataLoader worker

    This ensures that each worker process has a different random seed,
    preventing duplicated sampling in multiprocessing mode while still
    maintaining reproducibility.

    The worker seed is derived from PyTorch's initial seed (which is set
    by torch.manual_seed in the main process), combined with the worker ID.

    Args:
        worker_id: Worker process ID (0 to num_workers-1)
    """
    import numpy as np
    import random
    import torch

    # Get the initial seed from PyTorch (set in main process)
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(
    config,
    split="train",
    batch_size=8,
    num_workers=4,
    streaming=False,
    num_samples=None,
    dataset_name="wikitext",
    dataset_config="wikitext-103-raw-v1",
    primary_dataset_only=True,
    collate_fn=None
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
        dataset_name: Hugging Face dataset name
        dataset_config: Dataset configuration/version
        primary_dataset_only: If True, only use specified dataset without fallback
        collate_fn: Optional custom collate function for batching

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
            split=split,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            primary_dataset_only=primary_dataset_only
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
            num_samples=num_samples,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            primary_dataset_only=primary_dataset_only
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn  # Ensure reproducible worker seeds
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
