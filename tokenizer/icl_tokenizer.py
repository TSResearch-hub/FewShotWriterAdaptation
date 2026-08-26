import torch
import re
import numpy as np
import random

class ICLTokenizer():
    def __init__(self, icl_vocab_size, ooc_char, is_fixed_punctuations, mapping='relative'):

        self.mapping = mapping
        self.icl_vocab_size = icl_vocab_size
        self.ooc_char = ooc_char
        self.is_fixed_punctuations = is_fixed_punctuations
        self.bos_token = '<start>'
        self.eos_token = '<end>'
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.boc_token_id = 3
        self.backslash_token_id = 4
        self.ooc_token_id = 5
        self.mask_token_id = 6
        self.ctc_blank_token_id = 7

        # Initialize the special tokens, with <pad> and <ooc> first
        self.special_tokens = [
            '<pad>',
            '<start>',
            '<end>',
            '<context>',
            '<ⓝ>',
            '<ooc>',  # Add the out-of-context token
            # '<Ⓜ>',  # Add the masking token
            # '<ⓒ>' # Add the blank token for ctc

        ]
        self.special_token_ids = []
        # Build the vocabulary from N (starting after <pad> and <ooc>)
        self.vocab = {}
        token_index = 0
        # Add the special tokens
        for token in self.special_tokens:
            self.vocab[token] = token_index
            self.special_token_ids.append(token_index)
            token_index += 1

        # Add the tokens <t0>, <t1>, ..., <tN>
        for i in range(icl_vocab_size):
            self.vocab[f'<t{i}>'] = token_index
            token_index += 1

        # Add the punctuation tokens as unique tokens (e.g.: '<,>')
        # NOTE: only add SPACE
        self.space_token_id = token_index
        self.vocab[f'< >'] = token_index
        token_index += 1
        if self.is_fixed_punctuations:

            self.fixed_punctuations = list(",;?.:/-_!()[]")
            for punct in self.fixed_punctuations:
                self.vocab[f'<{punct}>'] = token_index
                token_index += 1

        # Build an inverted dictionary for text reconstruction
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        #definition of a dictionary with some fixed classes
        # it is used for encoding and decoding by the ContextAwareTokenizer
        default_reference_dict = {}
        default_reference_dict[self.ooc_char] = '<ooc>'
        default_reference_dict['ⓝ'] = '<ⓝ>'
        default_reference_dict[' '] = '< >'

        # Pre-fill with the fixed punctuations
        if self.is_fixed_punctuations:
            for punct in self.fixed_punctuations:
                default_reference_dict[punct] = f'<{punct}>'

        self.default_reference_dict = default_reference_dict


    def __len__(self):
        return len(self.vocab)

    def encode_one_token(self, text):
        return self.vocab[text]

    def encode_one_context(self, text):
        if self.mapping == 'relative':
            return self.encode_one_context_relative_pos(text)
        elif self.mapping == 'random':
            return self.encode_one_context_random(text)



    def encode_one_context_random(self, text):
        """Encoding specific to a character string:
        turns each character into a unique token.
        """
        # Copy of the default reference dictionary
        reference_dict = dict(self.default_reference_dict)

        # List of available token_ids
        possible_ids = list(range(0, self.icl_vocab_size))

        for char in text:
            if char not in reference_dict:
                if not possible_ids:
                    raise ValueError("No more token_ids available for the bijective mapping.")

                # Random selection of a token_id
                token_index = random.choice(possible_ids)
                possible_ids.remove(token_index)

                reference_dict[char] = f"<t{token_index}>"

        # Text encoding
        encoded_text = ''.join(reference_dict[char] for char in text)

        return encoded_text, reference_dict
    def encode_one_context_relative_pos(self, text):
        """Encoding specific to a character string: turns each character into a unique token."""
        reference_dict = {}  # Dictionary to keep the order of the first characters encountered
        reference_dict = reference_dict | self.default_reference_dict

        token_index = 0
        for char in text:
            if char not in reference_dict:
                # If the character hasn't been seen yet, assign it a new token
                reference_dict[char] = f'<t{token_index}>'
                token_index += 1

        # Use the unique_characters dictionary to encode the text
        encoded_text = ''.join([reference_dict[char] for char in text])

        return encoded_text, reference_dict


    def encode_contexts_into_positions(self, context_texts):
        """
        Encodes a batch of lists of texts, adding <pad> tokens for the shorter strings.

        Arguments:
            context_texts (list of list of str): List of lists, each sub-list contains texts to encode.

        Returns:
            list of list of str: List of encoded texts with padding added for each sub-list.
            dict: Dictionary mapping characters to tokens.
        """
        encoded_batch = []
        reference_dicts = []

        for text in context_texts:
            encoded, reference_dict = self.encode_one_context(text)
            encoded_batch.append(encoded)
            reference_dicts.append(reference_dict)

        return encoded_batch, reference_dicts


    def encode_one_target(self, text, reference_dict):
        """Encodes a text using a reference dictionary, with <ooc> for unknown characters."""
        return ''.join([reference_dict.get(char, '<ooc>') for char in text])

    def encode_targets_into_positions(self, target_texts, reference_dicts):
        encoded_texts = []
        for text, dict in zip(target_texts, reference_dicts):
            encoded_texts.append(self.encode_one_target(text, dict))
        return encoded_texts

    def pad(self, batch, pad_dir='left'):
        """
        Adds pad_token_id on the left of the sub-lists so that every sub-list has the same length.

        Arguments:
            batch (list of list of int): List of encoded sequences.
            pad_token_id (int): ID of the padding token to use (default 0).

        Returns:
            list of list of int: Batch with padding added.
        """
        # Find the maximum length across all sequences
        max_length = max(len(seq) for seq in batch)
        # Add padding tokens on the left for each shorter sequence
        if pad_dir=='left':
            return [[self.pad_token_id] * (max_length - len(seq)) + seq for seq in batch]
        if pad_dir=='right':
            return [seq + [self.pad_token_id] * (max_length - len(seq)) for seq in batch]

    def encode_icl_inputs(self, batch_text, pad_dir='left'):
        """Encoding with specific tokens: turns a string of tokens into indices."""
        encoded_batch = []

        for text in batch_text:
            # Split the text into tokens on the < and > tags, ignoring empty parts
            tokens = [f"{token}>" for token in text.split('>')][:-1]

            # Convert the valid tokens into indices
            encoded = [self.vocab[token] for token in tokens]

            encoded_batch.append(encoded)

        encoded_batch = self.pad(encoded_batch, pad_dir=pad_dir)
        return torch.tensor(encoded_batch, dtype=torch.long)

    def decode_custom_one_list(self, encoded_tokens, reference_dict):
        """
        Decoding of an encoded text: turns the tokens back into their original characters.

        Arguments:
            encoded_text (str): Text encoded with unique tokens.
            unique_characters (dict): Dictionary mapping tokens to characters.

        Returns:
            str: Decoded text.
        """

        # Invert the dictionary to map tokens to characters
        token_to_char = {v: k for k, v in reference_dict.items()}
        encoded_tokens = [f"{token}>" for token in encoded_tokens.split('>')][:-1]
        #encoded_tokens = [token.replace(' ', '') for token in encoded_tokens]
        encoded_tokens = [re.sub(r'\s+(<[^>]+>)', r'\1', token) for token in encoded_tokens]
        #decoded_text = ''.join([token_to_char[token] for token in encoded_tokens.split()])
        #decoded_text = ''.join([token_to_char.get(token, 'ⓔ') for token in encoded_tokens.split()])
        decoded_text = ''.join([token_to_char.get(token, 'ⓔ') for token in encoded_tokens])

        return decoded_text


    def decode_custom(self, batch_encoded, reference_dicts):
        """
        Decodes a batch of encoded texts using a dictionary mapping tokens to characters.

        Arguments:
            batch_encoded (list of list of str): List of batches, each sub-list contains encoded tokens.
            unique_dict (dict): Dictionary mapping tokens to characters.

        Returns:
            list of str: List of decoded texts.
        """
        decoded_batch = []

        for tokens, reference_dict in zip(batch_encoded, reference_dicts):
            # Decode each text in the batch using the unique_dict dictionary
            decoded_text = self.decode_custom_one_list(tokens, reference_dict)

            # Add the decoded text to the batch
            decoded_batch.append(decoded_text)

        return decoded_batch

    def decode(self, batch_encoded, ignore_special_tokens=True):
        """Decoding of a batch of encoded texts: turns indices back into tokens, with an exception for <ooc>."""

        # If ignore_special_tokens is True, define a list of special tokens to ignore
        if ignore_special_tokens:
            special_tokens = set(self.special_tokens)  - {"<ooc>"}  # Remove <ooc> from the set
        else:
            special_tokens = set()

        decoded_batch = []

        for encoded_text in batch_encoded:
            # Decode each text in the batch using the inverted vocabulary
            decoded_text = []

            for idx in encoded_text:
                token = self.inv_vocab.get(int(idx), None)
                if token and token not in special_tokens:
                    decoded_text.append(token)

            # Join the tokens to form the decoded text
            decoded_batch.append(' '.join(decoded_text))

        return decoded_batch

    def get_vocab(self):
        """Returns the full vocabulary as a dictionary."""
        return self.vocab



    def prepare_text_inputs(self, context_texts, target_texts, pad_dir_ctxt = 'left', pad_dir_tgt='right'):

        # Tokenize context and target texts
        context_tokens, reference_dicts = self.encode_contexts_into_positions(context_texts)
        target_tokens = self.encode_targets_into_positions(target_texts, reference_dicts)
        final_labels = self.decode_custom(target_tokens, reference_dicts)

        context_tokens = ['<context>' + tokens for tokens in context_tokens]
        target_tokens = [tokens + '<end>' for tokens in target_tokens]

        context_ids = self.encode_icl_inputs(context_tokens, pad_dir=pad_dir_ctxt)
        target_ids = self.encode_icl_inputs(target_tokens, pad_dir=pad_dir_tgt)



        return {
                "ctxt_ids": context_ids,
                "label_ids": target_ids,
                "label_strs": final_labels,
                "reference_dicts": reference_dicts,
            }



    def decode_output_ids(self, outputs, reference_dicts, mask=None, before='<end>'):
        if not isinstance(outputs, torch.Tensor):
            outputs = torch.tensor(outputs)
        if mask is not None:
            outputs = outputs * mask

        #outputs = outputs[:, :-1]
        img_end_token_id = self.vocab[before]
        new_outputs = []
        for ids in outputs:
            if img_end_token_id in ids:
                eos_locations = np.where(ids == img_end_token_id)[0]
                fist_pos = eos_locations[0]
                new_outputs.append(ids[:fist_pos])
            else:
                new_outputs.append(ids)
        outputs = new_outputs

        #outputs = [ids[:np.where(ids == img_end_token_id)[0][-1]] if img_end_token_id in ids else ids for ids in outputs]
        outputs = [seq[seq != 0].tolist() for seq in outputs]

        outputs = self.decode(outputs)
        outputs = self.decode_custom(outputs, reference_dicts=reference_dicts)
        return outputs
