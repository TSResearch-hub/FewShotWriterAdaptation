import torch
import re
import numpy as np
import random
from tokenizer.ocr_tokenizer import OCRTokenizer
from tokenizer.icl_tokenizer import ICLTokenizer

class GenericTokenizers:
    def __init__(self, config, ooc_char='*'):
        self.config = config
        data_config = config.data_config
        model_config = config.model_config
        self.training_tasks = model_config.training_tasks

        self.tokenizers = {}

        # if data_config.real_dataset == "IAM":
        #     self.charset = data_config.charset_iam
        # elif data_config.real_dataset == "RIMES":
        #     self.charset = config.data_config.charset_rimes
        # elif data_config.real_dataset == "UKR":
        #     if "ocr" in self.training_tasks:
        #         self.charset = config.data_config.charset_iam
        #     # elif "icl" in self.training_tasks:
        #     #     self.charset = config.data_config.charset_iam
        # elif data_config.real_dataset == "HKR":
        #     self.charset = config.data_config.charset_hkr
        # elif data_config.real_dataset == "CVL":
        #     self.charset = config.data_config.charset_iam

        # elif data_config.real_dataset == "BNF":
        #     self.charset = config.data_config.charset_bnf

        self.charset = data_config.charset

        #if "ocr" in self.training_tasks:
        self.tokenizers["ocr"] = OCRTokenizer(charset=self.charset)
        #alias added for evaluation
        self.decoder_ocr = self.tokenizers["ocr"]
        if "icl" in self.training_tasks:
            vocab_size = model_config.decoders.icl.vocab_size
            mapping = model_config.decoders.icl.mapping
            # if "omniglot" in data_config.synth_datasets:
            #     is_fixed_punctuations = False # fix ',.()!?
            # else:
            is_fixed_punctuations = True # fix ',.()!?

            self.tokenizers["icl"]  = ICLTokenizer(icl_vocab_size=vocab_size,
                                              ooc_char=ooc_char,
                                              is_fixed_punctuations=is_fixed_punctuations,
                                              mapping=mapping)

            #alias added for evaluation
            self.decoder_icl = self.tokenizers["icl"]



    def __getitem__(self, task):
        return self.tokenizers[task]

    def prepare_text_inputs(self, context_texts, target_texts, max_error_rate=0.2):
        inputs = {}
        tasks = list(set(self.training_tasks) | {"ocr"})
        for task in tasks:
            inputs[task] = {
                "ctxt_ids": None,
                "label_ids": None,
                "label_strs": None,
                "reference_dicts": None,
                "input_ids": None,
            }
            decoder_cfg = getattr(self.config.model_config.decoders, task)

            tokenized_inputs = self.tokenizers[task].prepare_text_inputs(
                                    context_texts=context_texts,
                                    target_texts=target_texts)


            # --- Add to the input dict --
            for key, value in tokenized_inputs.items():
                if key in set(inputs[task].keys()):
                    inputs[task][key] = value


            # == ADD START_TOKEN_ID and SHIFTING == #
            if decoder_cfg.is_autoreg:
                inputs[task]["input_ids"] = self.shift_right_and_add_start_token(
                                            labels=inputs[task]["label_ids"],
                                            start_token_id=self.tokenizers[task].bos_token_id)

            else:
                #initialize an input made only of start tokens
                if inputs[task]["input_ids"] is None :
                    #max_len = 128 #see config
                    inputs[task]["input_ids"] = torch.full_like(inputs[task]["label_ids"], self.tokenizers[task].bos_token_id)

                else :
                    # for the count task, the input is already prepared.
                    inputs[task]["input_ids"] = inputs[task]["input_ids"].contiguous()

                inputs[task]["label_ids"] = inputs[task]["label_ids"].contiguous()
                # batch_size = inputs[task]["input_ids"].size(0)
                # start_tokens = torch.full(
                #     (batch_size, 1),
                #     self.tokenizers[task].bos_token_id)
                # inputs[task]["input_ids"] = torch.cat([start_tokens, inputs[task]["input_ids"]], dim=1)
                # inputs[task]["label_ids"] = torch.cat([start_tokens, inputs[task]["label_ids"]], dim=1)
                #pass

            # == ADD NOISE in INPUT_IDS == #
            if task == "icl":
                inputs[task]["input_ids"] = self.add_noise_to_input_ids_icl(
                                                context_ids=inputs[task]["ctxt_ids"],
                                                input_ids=inputs[task]["input_ids"],
                                                max_error_rate=max_error_rate)
            elif task == "ocr":
                inputs[task]["input_ids"] = self.add_noise_to_input_ids_ocr(
                                                input_ids=inputs[task]["input_ids"],
                                                max_error_rate=max_error_rate)

        return inputs


    def shift_right_and_add_start_token(self, labels, start_token_id=0):
        """
        Shifts the labels to the right to serve as decoder input.
        Inserts a start token (here start_token_id or another).
        """
        shifted_input_ids = labels.new_zeros(labels.size())
        shifted_input_ids[:, 1:] = labels[:, :-1]
        shifted_input_ids[:, 0] = start_token_id  # or a special <bos> if you have one
        return shifted_input_ids


    def add_noise_to_input_ids_icl(
        self,
        context_ids,
        input_ids,
        max_error_rate=0.2,
        max_actual_error_rate=0.5  # safety threshold
    ):
        """
        Adds controlled lexical noise into input_ids by replacing certain tokens
        with tokens taken from the context, with control over the effective error rate.

        Logic:
        1. 0 to 2 grouped replacements (all identical tokens replaced)
        2. Local errors to fill up to num_errors
        3. Compute the actual error rate
        4. Safety block: if the rate exceeds max_actual_error_rate, restore
        some modified tokens to lower the rate.
        """

        tokenizer = self.tokenizers["icl"]
        pad_id = tokenizer.pad_token_id
        forbidden_ids = tokenizer.special_token_ids + [tokenizer.space_token_id]

        noisy_inputs = input_ids.clone()
        actual_error_rates = []

        bsz, seq_len = input_ids.shape

        for b in range(bsz):
            labels_seq = noisy_inputs[b].clone()
            original_seq = labels_seq.clone()

            # Valid positions (outside padding & forbidden)
            valid_positions = [
                i for i in range(seq_len)
                if labels_seq[i].item() != pad_id and labels_seq[i].item() not in forbidden_ids
            ]
            eff_len = len(valid_positions)
            if eff_len == 0:
                actual_error_rates.append(0.0)
                continue

            # Usable context
            context_seq = [
                t for t in context_ids[b].tolist() if t != pad_id and t not in forbidden_ids
            ]
            if not context_seq:
                actual_error_rates.append(0.0)
                continue

            # Total number of errors to inject
            error_rate = random.uniform(0.0, max_error_rate)
            num_errors = max(1, int(error_rate * eff_len))
            remaining_errors = num_errors

            # ------------------------------
            # 1️⃣ Grouped noise (0 to 2 replacements)
            # ------------------------------
            num_grouped_replacements = random.randint(0, 2)
            for _ in range(num_grouped_replacements):
                token_counts = {}
                for i in valid_positions:
                    token = labels_seq[i].item()
                    if token not in forbidden_ids:
                        token_counts[token] = token_counts.get(token, 0) + 1

                repeated_tokens = [t for t, c in token_counts.items() if c > 1]
                if not repeated_tokens:
                    break

                token_to_replace = random.choice(repeated_tokens)

                context_counts = {}
                for t in context_seq:
                    context_counts[t] = context_counts.get(t, 0) + 1
                repeated_context_tokens = [t for t, c in context_counts.items() if c > 1]
                if not repeated_context_tokens:
                    repeated_context_tokens = context_seq  # fallback

                replacement_token = random.choice(repeated_context_tokens)

                # Global replacement
                for i in valid_positions:
                    if labels_seq[i].item() == token_to_replace:
                        labels_seq[i] = replacement_token

            # ------------------------------
            # 2️⃣ Local noise
            # ------------------------------
            if remaining_errors > 0:
                positions_for_local = [
                    i for i in valid_positions if labels_seq[i].item() not in forbidden_ids
                ]
                if positions_for_local:
                    positions_to_replace = random.sample(
                        positions_for_local, min(remaining_errors, len(positions_for_local))
                    )
                    for pos in positions_to_replace:
                        original = labels_seq[pos].item()
                        replacement_choices = list(set(context_seq) - {original})
                        if replacement_choices:
                            labels_seq[pos] = random.choice(replacement_choices)

            # ------------------------------
            # 3️⃣ Compute the actual error rate
            # ------------------------------
            diff_positions = [i for i in valid_positions if labels_seq[i].item() != original_seq[i].item()]
            actual_error_rate = len(diff_positions) / eff_len

            # ------------------------------
            # 4️⃣ Safety block: cap the error rate at max_actual_error_rate
            # ------------------------------
            if actual_error_rate > max_actual_error_rate:
                # number of tokens to restore
                num_to_restore = int((actual_error_rate - max_actual_error_rate) * eff_len)
                restore_positions = random.sample(diff_positions, min(num_to_restore, len(diff_positions)))
                for i in restore_positions:
                    labels_seq[i] = original_seq[i]

                # recompute the actual error rate
                diff_positions = [i for i in valid_positions if labels_seq[i].item() != original_seq[i].item()]
                actual_error_rate = len(diff_positions) / eff_len

            noisy_inputs[b] = labels_seq
            actual_error_rates.append(actual_error_rate)
        #print(np.mean(actual_error_rates))
        return noisy_inputs


    def add_noise_to_input_ids_ocr(self, input_ids, max_error_rate=0.2):
        """
        Adds noise into input_ids (shape [bsz, seq_len])
        by replacing certain ids with others drawn from the full charset.

        - Noise is always applied
        - Noise proportion ∈ [0, max_error_rate] per sequence
        - No grouped/local mode
        - Does not touch special tokens
        """

        forbidden_ids = self.tokenizers["ocr"].special_token_ids
        pad_token_id = self.tokenizers["ocr"].pad_token_id
        noisy_input_ids = input_ids.clone()
        bsz, seq_len = input_ids.shape
        # list of ids allowed for replacement
        vocab_size = len(self.tokenizers["ocr"])
        valid_ids = [i for i in range(vocab_size) if i not in forbidden_ids]

        for b in range(bsz):
            labels_seq = noisy_input_ids[b]
            seq_len = len(labels_seq[labels_seq != pad_token_id])

            if not valid_ids:
                continue
            # noise proportion drawn at random
            error_rate = random.uniform(0.0, max_error_rate)
            num_errors = max(0, int(error_rate * seq_len))

            # allowed positions (≠ forbidden_ids)
            valid_positions = [i for i in range(seq_len) if labels_seq[i].item() not in forbidden_ids]
            if not valid_positions:
                continue
            # select positions to add noise to
            positions = random.sample(valid_positions, min(num_errors, len(valid_positions)))

            for pos in positions:
                original = labels_seq[pos].item()
                replacement_choices = [i for i in valid_ids if i != original]
                if not replacement_choices:
                    continue
                labels_seq[pos] = random.choice(replacement_choices)

            noisy_input_ids[b] = labels_seq

        return noisy_input_ids
