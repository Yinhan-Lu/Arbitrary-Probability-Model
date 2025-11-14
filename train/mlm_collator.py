import torch
import math

class MLMDataCollator:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id

        # NEW: handle tokenizers that may not have CLS/SEP (e.g., GPT-2)
        self.cls_token_id = getattr(tokenizer, "cls_token_id", None)
        self.sep_token_id = getattr(tokenizer, "sep_token_id", None)

        # Optional safety: make sure we actually have a [MASK] token
        if self.mask_token_id is None:
            raise ValueError(
                "Tokenizer has no mask_token_id. "
                "Add a [MASK] token before creating MLMDataCollator."
            )

    def __call__(self, batch_examples):
        # batch_examples: list of dicts with "input_ids"
        input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch_examples]
        batch = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
        )  # (B, T)

        labels = batch.clone()

        # create mask
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # start with pad tokens
        special_mask = (batch == self.pad_token_id)
        # only add CLS/SEP if the tokenizer has them
        if self.cls_token_id is not None:
            special_mask |= (batch == self.cls_token_id)
        if self.sep_token_id is not None:
            special_mask |= (batch == self.sep_token_id)

        probability_matrix.masked_fill_(special_mask, 0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # we only predict masked positions

        # 80% [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        batch[indices_replaced] = self.mask_token_id

        # 10% random token
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices & ~indices_replaced
        )
        random_words = torch.randint(
            low=0,
            high=self.tokenizer.vocab_size,
            size=labels.shape,
            dtype=torch.long,
        )
        batch[indices_random] = random_words[indices_random]

        # remaining 10%: keep original token but still predict it

        # attention mask: 1 for real tokens, 0 for pad
        attention_mask = (batch != self.pad_token_id).long()

        return {
            "input_ids": batch,
            "attention_mask": attention_mask,
            "labels": labels,
        }
