import torch
import numpy as np
from typing import List

class OCRTokenizer():
    def __init__(self, charset):
        self.charset = charset

        # Token strings
        self.bos_token   = '<start>'
        self.eos_token   = '<end>'
        self.pad_token   = '<pad>'
        self.unk_token   = '<unk>'
        self.context_token = '<context>'
        self.backslash_token = 'ⓝ'
        self.mask_token  = '<Ⓜ>'
        self.ctc_blank_token = '<ⓒ>'

        # Token IDs — fixed order, modeled on CharTokenizer (PAD=0, BOS=1, EOS=2, …)
        self.pad_token_id        = 0
        self.bos_token_id        = 1
        self.eos_token_id        = 2
        self.unk_token_id        = 3
        self.boc_token_id        = 4   # <context>
        self.backslash_token_id  = 5   # ⓝ
        self.mask_token_id       = 6   # <Ⓜ>
        self.ctc_blank_token_id  = 7   # <ⓒ>

        self.special_tokens = [
            '<pad>',      # 0
            '<start>',    # 1
            '<end>',      # 2
            '<unk>',      # 3
            '<context>',  # 4
            'ⓝ',          # 5
            '<Ⓜ>',        # 6
            '<ⓒ>',        # 7
        ]

        # Vocabulary construction
        self.special_token_ids = []
        self.vocab = {}
        for idx, token in enumerate(self.special_tokens):
            self.vocab[token] = idx
            self.special_token_ids.append(idx)


        # Adding the charset characters
        for idx, char in enumerate(self.charset):
            self.vocab[char] = idx + len(self.special_tokens)

        # Inverted vocabulary
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        print(f'VOCAB SIZE TOKENIZER : {self.vocab_size}')

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        return self.vocab_size

    # ------------------------------------------------------------------
    # Encode / decode (character level, modeled on CharTokenizer)
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Encode a string into a list of IDs (without BOS/EOS)."""
        return [
            self.vocab.get(char, self.unk_token_id)
            for char in text
            if char in self.vocab
        ]

    def decode(self, token_ids: List[int], ignore_special_tokens: bool = True) -> str:
        """
        Decode a list of IDs into a string.
        Stops at the first EOS; ignores PAD and BOS by default.
        """
        result = []
        skip_ids = {self.pad_token_id, self.bos_token_id} if ignore_special_tokens else set()

        for token_id in token_ids:
            token_id = int(token_id)
            # Stop at the EOS token
            if token_id == self.eos_token_id:
                break
            # CTC blank filter
            if token_id == self.ctc_blank_token_id:
                continue
            if token_id in skip_ids:
                continue
            token = self.ids_to_tokens.get(token_id)
            if token is not None:
                result.append(token)

        return "".join(result)

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def _pad_batch(self, batch: List[List[int]], pad_dir: str = 'right') -> List[List[int]]:
        """Pad all sequences to the maximum length of the batch."""
        max_length = max(len(seq) for seq in batch)
        if pad_dir == 'left':
            return [[self.pad_token_id] * (max_length - len(seq)) + seq for seq in batch]
        return [seq + [self.pad_token_id] * (max_length - len(seq)) for seq in batch]

    # ------------------------------------------------------------------
    # Batch-oriented encoding (original methods kept)
    # ------------------------------------------------------------------

    def encode_contexts(self, batch_text: List[List[str]]) -> torch.Tensor:
        """
        Encode a batch of contexts.
        Format: [<context>, char1, char2, …] — left-padded.
        """
        encoded_batch = []
        for text in batch_text:
            encoded = [self.boc_token_id]
            for token in text:
                encoded.extend(self.encode(token))
            encoded_batch.append(encoded)

        encoded_batch = self._pad_batch(encoded_batch, pad_dir='left')
        return torch.tensor(encoded_batch, dtype=torch.long)

    def encode_inputs(self, batch_text: List[str]) -> torch.Tensor:
        """
        Encode a batch of target labels.
        Format: [char1, char2, …, <end>] — right-padded.
        """
        encoded_batch = []
        for text in batch_text:
            encoded = self.encode(text)
            encoded.append(self.eos_token_id)
            encoded_batch.append(encoded)

        encoded_batch = self._pad_batch(encoded_batch, pad_dir='right')
        return torch.tensor(encoded_batch, dtype=torch.long)

    def prepare_text_inputs(
        self,
        context_texts: List[List[str]] | None,
        target_texts: List[str],
    ) -> dict:
        try:
            label_ids = self.encode_inputs(target_texts)
        except Exception:
            label_ids = torch.zeros((1, 1), dtype=torch.long)

        return {
            "ctxt_ids":   self.encode_contexts(context_texts) if context_texts is not None else None,
            "label_ids":  label_ids,
            "label_strs": target_texts,
        }

    # ------------------------------------------------------------------
    # Batch decoding (model output)
    # ------------------------------------------------------------------

    def decode_batch(
        self,
        batch_encoded: List[List[int]],
        ignore_special_tokens: bool = True,
    ) -> List[str]:
        """Decode a batch of ID sequences."""
        return [self.decode(seq, ignore_special_tokens) for seq in batch_encoded]

    def decode_output_ids(
        self,
        outputs: List,
        before: str = '<end>',
    ) -> List[str]:
        """
        Decode the model outputs:
        - cuts before the `before` token
        - removes residual PADs
        """
        cut_token_id = self.vocab[before]
        cleaned = []

        for ids in outputs:
            ids = ids.detach().cpu().numpy() if hasattr(ids, 'detach') else np.array(ids)

            # Cut before the target token
            positions = np.where(ids == cut_token_id)[0]
            if len(positions):
                ids = ids[:positions[0]]

            # Remove residual PADs
            ids = ids[ids != self.pad_token_id]
            cleaned.append(ids.tolist())

        return self.decode_batch(cleaned)

    # ------------------------------------------------------------------
    # Vocabulary access
    # ------------------------------------------------------------------

    def encode_one_token(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def get_vocab(self) -> dict:
        return self.vocab
