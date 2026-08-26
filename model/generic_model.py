import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerEncoder, TransformerEncoderLayer
from torch.nn.init import trunc_normal_
import torch.nn.functional as F
import math
from typing import Optional
from model.fcn_encoder import FCN_Encoder
from torch import Tensor

class TransformerDecoderLayerWithAttn(nn.TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_attn_weights = None
        self.cross_attn_weights = None

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:

        # Self-attention
        tgt2, self_attn = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        self.self_attn_weights = self_attn
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention
        tgt2, cross_attn = self.multihead_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        self.cross_attn_weights = cross_attn
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoderWrapper(nn.Module):
    def __init__(self, d_model=256, dim_feedforward=512, nhead=2, num_layers=2):
        super().__init__()
        
        #Definition of the standard decoder
        # decoder_layer = TransformerDecoderLayer(
        #     d_model=d_model,
        #     nhead=nhead,
        #     dim_feedforward=dim_feedforward,
        #     batch_first=True
        # )

        decoder_layer = TransformerDecoderLayerWithAttn(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )

        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
       
    def forward(
        self,
        tgt_emb: torch.Tensor,                           # (B, T, d_model)
        memory: torch.Tensor,                            # (B, S, d_model)
        tgt_mask: Optional[torch.Tensor] = None,         # (T, T)
        memory_mask: Optional[torch.Tensor] = None,      # (T, S)
        tgt_key_padding_mask: Optional[torch.Tensor] = None,      # (B, T)
        memory_key_padding_mask: Optional[torch.Tensor] = None,   # (B, S)
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            tgt_emb: Embedded target tokens (B, T, d_model)
            memory: Encoder output (B, S, d_model)
            tgt_mask: Causal or custom mask for decoder self-attention (T, T)
            memory_mask: Optional mask for encoder-decoder attention (T, S)
            tgt_key_padding_mask: Padding mask for tgt (B, T), True for PAD
            memory_key_padding_mask: Padding mask for memory (B, S), True for PAD
            tgt_is_causal: Hint that tgt_mask is a causal mask
            memory_is_causal: Hint that memory_mask is causal
        Returns:
            Logits: (B, T, vocab_size)
        """
       
        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal,
        )
        return out #self.out_proj(out)  # (B, T, vocab_size)


class SinusoidalPositionalEncoding1D(nn.Module):
    def __init__(self, dim_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_model, 2) * (-math.log(10000.0) / dim_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, pos_ids):
        return self.pe[pos_ids]


class GenericModel(nn.Module):
    def __init__(self, model_config, tokenizers):
        super().__init__()

        self.model_config = model_config
        dim_model = model_config.encoder.dim_model
        
        self.encoder = FCN_Encoder(output_dim=dim_model,
                                init_dropout_rate=model_config.encoder.start_dropout_rate,
                                )
            

        self.max_seq_len = model_config.max_seq_len
        self.training_tasks = model_config.training_tasks
        
        if model_config.pos_enc == "static":
            self.pos_encoding = SinusoidalPositionalEncoding1D(dim_model=dim_model,
                                                            max_len=self.max_seq_len)
        else:
            self.pos_encoding = nn.Embedding(self.max_seq_len, dim_model, padding_idx=0)

       
        
        # ==========================
        # MODEL INITIALIZATION
        # ==========================

        self.embeddings = nn.ModuleDict()
        self.losses_fn = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.out_projs = nn.ModuleDict()
        self.is_autoregs = {}
        self.with_contexts = {}
        self.tokenizers = tokenizers
        

        if model_config.is_cnn_ctc:
            blank_id = self.tokenizers["ocr"].ctc_blank_token_id
            self.losses_fn["ocr"] = nn.CTCLoss(
                        blank=blank_id,
                        reduction="mean",
                        zero_infinity=True)

            vocab_size = len(tokenizers["ocr"])
            self.out_projs["ocr"] = nn.Linear(dim_model, vocab_size)

        for task in self.training_tasks:  # e.g.: 'icl', 'ocr'

            vocab_size = len(tokenizers[task])

            # === Embedding ===
            self.embeddings[task] = nn.Embedding(vocab_size, dim_model, padding_idx=tokenizers[task].pad_token_id)
            nn.init.normal_(self.embeddings[task].weight, mean=0.0, std=dim_model ** -0.5)

            # === Loss function ===
            pad_token_id = tokenizers[task].pad_token_id

            # === Decoder configuration ===
            decoder_config = getattr(model_config.decoders, task)
            self.with_contexts[task] = decoder_config.with_context
            self.is_autoregs[task] = decoder_config.is_autoreg

            if self.is_autoregs[task]:
                # if not autoregressive, use a CTC loss instead
                if pad_token_id is not None :
                    self.losses_fn[task] = nn.CrossEntropyLoss(ignore_index=pad_token_id)
                else:
                    self.losses_fn[task] = nn.CrossEntropyLoss()
            else:
                # -------------------------------------------------
                # PARALLEL MODE → CTC
                # -------------------------------------------------
                blank_id = self.tokenizers[task].ctc_blank_token_id  # or a dedicated blank token
                self.losses_fn[task] = nn.CTCLoss(
                    blank=blank_id,
                    reduction="mean",
                    zero_infinity=True,  # important for numerical stability
                )

            # === Transformer Decoder ===
            self.decoders[task] = TransformerDecoderWrapper(
                d_model=dim_model,
                nhead=decoder_config.num_heads,
                num_layers=decoder_config.num_layers,
                dim_feedforward=decoder_config.dim_ffwd,
            )

            # === Output projection (logits → vocab) ===
            self.out_projs[task] = nn.Linear(dim_model, vocab_size)
      

    def forward(self, inputs):
        outputs = {
            "logits": None, 
            "loss": None,
            "pad_label_ids":None
        }
        
        feat_map = inputs["fw_2D"]
        B, C, H, W = feat_map.shape
        feat_mask = inputs["fw_masks_2D"]
        del inputs["fw_2D"], inputs["fw_masks_2D"]

        if self.model_config.is_cnn_ctc:
            logits_ctc, input_lengths = self._run_ctc_decoder(fw_2D=feat_map, 
                                               fw_masks_2D=feat_mask)
                                            #    img_original_widths=inputs["query_img_original_widths"],
                                            #    stride=inputs["stride"])
            outputs["logits_ctc"] = logits_ctc
            outputs["loss_ctc"] = self._run_ctc_loss("cnnctc", 
                                                     logits_ctc, 
                                                     label_ids=inputs["ocr"]["label_ids"],
                                                     input_lengths=input_lengths,
                                                     )

        pos_enc = self.encoder.positional_encoding(H, W).to(feat_map.device)  # (1, H*W, d_model)
        feat_map = feat_map.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, d_model)
        feat_map = feat_map + pos_enc
         #  convert feat_mask to boolean with pad = 0 and content > 0
        feat_mask = (feat_mask == 0)  # True = padding, False = content
        feat_mask = feat_mask.view(B, 1, -1).permute(0, 2, 1).squeeze(-1) # (B, H*W, 1)

        if len(self.training_tasks) != 0:
            task = self.training_tasks[0]
            # returns input_embs, an embedding based on [ctxt, input, pad]
            memory, input_embs, mem_mask, input_masks, pad_label_ids = self._prepare_context_and_input(
                task=task,
                ctxt_ids=inputs[task]['ctxt_ids'],
                input_ids=inputs[task]['input_ids'],
                fw=feat_map,
                fw_masks=feat_mask,
                label_ids=inputs[task]["label_ids"]
            )

            outputs["pad_label_ids"] = pad_label_ids
            

            logits = self._run_decoder(
                            task=task,
                            input_embs=input_embs,
                            input_masks=input_masks,
                            memory=memory,
                            memory_key_padding_mask=mem_mask,
                        )
            
            outputs["logits"] = logits
            outputs["loss"] = self._run_loss(task, logits, label_ids=pad_label_ids)

        return outputs
    
    def _run_ctc_loss(self, task, logits, label_ids, input_lengths=[]):
        # (B, T, C) → (T, B, C)
        log_probs = logits.log_softmax(dim=-1)
        log_probs = log_probs.permute(1, 0, 2)

        B = logits.size(0)
        T = logits.size(1)

        if task == "cnnctc":
            task = "ocr"  # for the rest of the CTC loss code, we refer to "ocr" for tokenizers and other configs
        else:
            # Parallel Transformer Decoder (non-autoregressive)
            # → we assume the input sequence is already aligned and of length T for every sample in the batch
            input_lengths = torch.full(
                (B,),
                fill_value=T,
                dtype=torch.long,
                device=logits.device
            )

        pad_id = self.tokenizers[task].pad_token_id
        # Actual target length
        target_lengths = (label_ids != pad_id).sum(dim=1)

        if not (input_lengths >= 2 * target_lengths - 1).all():
            print("Warning: Some input lengths are too short for CTC. This may cause NaN losses.")

        # CTC expects concatenated targets
        targets = label_ids[label_ids != pad_id]
        loss = self.losses_fn[task](
            log_probs,
            targets,
            input_lengths,
            target_lengths
        )
        return loss

    def _right_pad_ids(self, ids, pad_id):
        """
        Moves padding from the left to the right.
        ids : (B, L)
        """
        B, L = ids.size()

        new_ids = torch.full_like(ids, pad_id)

        for b in range(B):
            valid = ids[b] != pad_id
            valid_len = valid.sum().item()

            if valid_len > 0:
                new_ids[b, :valid_len] = ids[b, valid]

        return new_ids
    
  
    def _prepare_context_and_input(
        self,
        task,
        ctxt_ids,
        input_ids,
        fw,
        fw_masks,
        label_ids,
    ):
        if self.is_autoregs[task]:
            return self._prepare_context_and_input_autoreg(task=task,
                                                        ctxt_ids=ctxt_ids,
                                                        input_ids=input_ids,
                                                        fw=fw,
                                                        fw_masks=fw_masks,
                                                        label_ids=label_ids)
        else:
             return self._prepare_context_and_input_ctc(task=task,
                                                        ctxt_ids=ctxt_ids,
                                                        fw=fw,
                                                        fw_masks=fw_masks,
                                                        label_ids=label_ids)
    def _prepare_context_and_input_autoreg(
        self,
        task,
        ctxt_ids,
        input_ids,
        fw,
        fw_masks,
        label_ids,
    ):
        """
        - concat IDs (ctxt + input)
        - globally shift padding to the right
        - embedding + positional encoding
        - label alignment: [pad_before_start, label_ids, pad]

        Returns:
            memory
            input_embs
            mem_mask
            input_mask
            pad_label_ids
        """
        B, device = fw.size(0), fw.device
        tokenizer = self.tokenizers[task]
        pad_id = tokenizer.pad_token_id
        start_id = tokenizer.bos_token_id  # <start>

        # ----- MEMORY -----
        if fw_masks is None:
            _, L_fw, _ = fw.size()
            fw_masks = torch.zeros((B, L_fw), dtype=torch.bool, device=device)

        memory = fw
        mem_mask = fw_masks

        if ctxt_ids is None and input_ids is None:
            return memory, None, mem_mask, None, None

        # ----- CONCAT IDS (BRUT) -----
        ids_parts = []

        if self.with_contexts.get(task, False) and ctxt_ids is not None:
            ids_parts.append(ctxt_ids)

        if input_ids is not None:
            ids_parts.append(input_ids)

        all_ids = torch.cat(ids_parts, dim=1)

        # ----- GLOBAL PADDING SHIFT -----
        all_ids = self._right_pad_ids(all_ids, pad_id)

        # ------------------------------------------------------------------
        # ----- BUILD ALIGNED LABELS: [pad, label, pad] ----
        # ------------------------------------------------------------------
        if label_ids is None:
            pad_label_ids = None
        else:
            B, L_total = all_ids.shape
            pad_label_ids = torch.full(
                (B, L_total),
                fill_value=pad_id,
                dtype=label_ids.dtype,
                device=device
            )

            # actual label lengths (without padding)
            label_lengths = (label_ids != pad_id).sum(dim=1)

            for b in range(B):
                # 🔑 find the <start> position
                start_positions = (all_ids[b] == start_id).nonzero(as_tuple=False)

                if start_positions.numel() == 0:
                    # safety: no <start> found → skip
                    continue

                start_idx = start_positions[0].item()  # first <start>

                l_len = label_lengths[b].item()
                if l_len > 0:
                    end_idx = min(start_idx + l_len, L_total)
                    pad_label_ids[b, start_idx:end_idx] = label_ids[b, : end_idx - start_idx]

        # ----- EMBEDDING + POSITIONAL ENCODING -----
        input_embs, input_mask, _ = self._embed_and_mask(
            all_ids,
            task,
            pos_offset=None
        )

       
        return memory, input_embs, mem_mask, input_mask, pad_label_ids

    
    def _prepare_context_and_input_ctc(
        self,
        task,
        ctxt_ids,
        fw,
        fw_masks,
        label_ids,
        max_seq_len=160,
    ):
        """
        Builds the decoder input in the form:
            [160 MASK TOKENS | CONTEXT | PAD]

        - The 160 mask tokens ALWAYS occupy positions 0..159
        - The context is shifted to the right
        - Global right-padding
        - CTC labels are aligned on the first 160 tokens

        Returns:
            memory, input_embs, mem_mask, input_mask, pad_label_ids
        """

        B, device = fw.size(0), fw.device
        tokenizer = self.tokenizers[task]

        pad_id = tokenizer.pad_token_id
        mask_id = tokenizer.mask_token_id

        # ---------------- MEMORY ----------------
        if fw_masks is None:
            _, L_fw, _ = fw.size()
            fw_masks = torch.zeros((B, L_fw), dtype=torch.bool, device=device)

        memory = fw
        mem_mask = fw_masks

        # ---------------- CONTEXT ----------------
        if self.with_contexts.get(task, False) and ctxt_ids is not None:
            context_ids = ctxt_ids
        else:
            context_ids = None

        # ---------------- 160 MASK TOKENS ----------------
        mask_tokens = torch.full(
            (B, max_seq_len),
            fill_value=mask_id,
            dtype=torch.long,
            device=device,
        )

        # ---------------- CONCAT : [MASK_160 | CONTEXT] ----------------
        if context_ids is not None:
            all_ids = torch.cat([mask_tokens, context_ids], dim=1)
        else:
            all_ids = mask_tokens

        # ---------------- GLOBAL RIGHT PADDING ----------------
        all_ids = self._right_pad_ids(all_ids, pad_id)

        # ---------------- ALIGNED LABELS (positions 0..159) ----------------
        if label_ids is None:
            pad_label_ids = None
        else:
            L_total = all_ids.size(1)

            pad_label_ids = torch.full(
                (B, L_total),
                fill_value=pad_id,
                dtype=label_ids.dtype,
                device=device,
            )

            for b in range(B):
                valid_labels = label_ids[b][label_ids[b] != pad_id][:max_seq_len]
                n = valid_labels.size(0)
                if n > 0:
                    pad_label_ids[b, :n] = valid_labels

        # ---------------- EMBEDDING + MASK ----------------
        input_embs, input_mask, _ = self._embed_and_mask(
            all_ids,
            task,
            pos_offset=None,
        )

        return memory, input_embs, mem_mask, input_mask, pad_label_ids


    def _embed_and_mask(self, ids, task, pos_offset=None):
        """
        ids: (B, L)
        pos_offset: (B,) or None
        """

        pad_id = self.tokenizers[task].pad_token_id
        mask = (ids == pad_id)  # True = PAD

        # ------------------------------------------------
        # BERT-STYLE POSITIONS (independent of padding)
        # PAD -> 0
        # real tokens -> 1..N in order
        # ------------------------------------------------
        # cumsum of real tokens
        pos = torch.cumsum(~mask, dim=1)

        # PAD = 0
        pos = pos.masked_fill(mask, 0)

        # ----- OFFSET (context -> input continuity) -----
        if pos_offset is not None and pos_offset != 0:
            pos = pos + pos_offset.unsqueeze(1)

        # ----- LAST REAL POSITION (per batch) -----
        last_pos = pos.masked_fill(mask, -1).max(dim=1).values

        # ----- EMBEDDINGS -----
        pos_emb = self.pos_encoding(pos)
        pos_emb = pos_emb.masked_fill(mask.unsqueeze(-1), 0.0)

        emb = self.embeddings[task](ids) + pos_emb

        return emb, mask, last_pos

    def _run_decoder(
        self,
        task,
        input_embs,
        memory,
        memory_key_padding_mask,
        input_masks=None,
        max_seq_len=160,
    ):
        device = input_embs.device
       
        # -------------------------------------------------
        # AUTOREGRESSIVE
        # -------------------------------------------------
        if self.is_autoregs[task]:
            L = input_embs.size(1)

            causal_mask = torch.triu(
                torch.ones((L, L), dtype=torch.bool, device=device),
                diagonal=1
            )

            hidden = self.decoders[task](
                tgt_emb=input_embs,
                memory=memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=input_masks,
                memory_mask=None,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=False,
            )

            logits = self.out_projs[task](hidden) # AR → whole sequence

        # -------------------------------------------------
        # CTC
        # -------------------------------------------------
        else : 
            hidden = self.decoders[task](
                tgt_emb=input_embs,
                tgt_key_padding_mask=input_masks,
                memory=memory,
                memory_mask=None,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=False,
            )
            logits = self.out_projs[task](hidden)
            # Extract only the Q tokens after the context
            #logits = logits[:, ctxt_len:ctxt_len + max_seq_len, :]

            # The 160 mask tokens are now at the head of the sequence
            logits = logits[:, :max_seq_len, :]

        return logits


    def _run_ctc_decoder(self, fw_2D, fw_masks_2D):

        # 0️⃣ Input normalization
        if fw_2D.is_sparse:
            fw_2D = fw_2D.to_dense()
        if fw_masks_2D.is_sparse:
            fw_masks_2D = fw_masks_2D.to_dense()
        if fw_masks_2D.dim() == 4:
            fw_masks_2D = fw_masks_2D.squeeze(1)  # (B, H, W)

        B, C, H, W = fw_2D.shape

        # 1️⃣ Query mask: value 3 = query content
        query_mask = (fw_masks_2D == 3)  # (B, H, W)

        # Valid columns = columns where AT LEAST one H pixel is query
        col_valid = query_mask.any(dim=1)  # (B, W)

        logits_list = []
        input_lengths = []

        for i in range(B):
            valid_cols = col_valid[i]  # (W,)

            # Safety: no query column
            if not valid_cols.any():
                vocab = self.out_projs["ocr"].out_features
                logits_list.append(torch.zeros(1, vocab, device=fw_2D.device))
                input_lengths.append(1)
                continue

            # 2️⃣ Extract only the query columns
            fw_valid   = fw_2D[i, :, :, valid_cols]             # (C, H, W_query)
            mask_valid = query_mask[i, :, valid_cols].float()   # (H, W_query) 1=query 0=context/padding/sep

            # 3️⃣ Masked mean pooling over H — average over query pixels only
            masked   = fw_valid * mask_valid.unsqueeze(0)                   # (C, H, W_query)
            mask_sum = mask_valid.sum(dim=0, keepdim=True).clamp(min=1e-6)  # (1, W_query)
            pooled   = masked.sum(dim=1) / mask_sum                         # (C, W_query)

            # 4️⃣ Projection → logits
            logits_i = self.out_projs["ocr"](pooled.T)  # (W_query, vocab)
            logits_list.append(logits_i)
            input_lengths.append(logits_i.shape[0])

        # 5️⃣ Re-pad to form a batch (B, T_max, vocab)
        T_max = max(input_lengths)
        vocab = logits_list[0].shape[-1]
        logits_padded = fw_2D.new_zeros(B, T_max, vocab)
        for i, logits_i in enumerate(logits_list):
            logits_padded[i, :logits_i.shape[0]] = logits_i

        input_lengths = torch.tensor(
            input_lengths,
            dtype=torch.long,
            device=fw_2D.device
        )

        return logits_padded, input_lengths

   
    def _run_loss(
            self,
            task,
            logits,
            label_ids,
        ):
        loss_fn = self.losses_fn[task]

        # CTC CASE
        if isinstance(loss_fn, nn.CTCLoss):
            loss = self._run_ctc_loss(task, logits, label_ids=label_ids)

        else:

            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                label_ids.view(-1)
            )

        return loss

    

    @torch.no_grad()
    def generate(self, inputs, max_len: int = 200, temperature: float = 0.0):
        """
        Sequence generation for each task (OCR, ICL, ...).

        Args:
            inputs: dictionary of inputs per task
            max_len (int): maximum generation length
            temperature (float): sampling temperature
        """

        outputs = {}
        feat_map = inputs["fw_2D"]
        B, C, H, W = feat_map.shape
        feat_mask = inputs["fw_masks_2D"]
        del inputs["fw_2D"], inputs["fw_masks_2D"]

        if self.model_config.is_cnn_ctc:
            logits_ctc, _ = self._run_ctc_decoder(fw_2D=feat_map, 
                                               fw_masks_2D=feat_mask)
                                            #    img_original_widths=inputs["query_img_original_widths"],
                                            #    stride=inputs["stride"])
            outputs["logits_ctc"] = logits_ctc


        pos_enc = self.encoder.positional_encoding(H, W).to(feat_map.device)  # (1, H*W, d_model)
        feat_map = feat_map.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, d_model)
        feat_map = feat_map + pos_enc

        #  convert feat_mask to boolean with pad = 0 and content > 0
        feat_mask = (feat_mask == 0)  # True = padding, False = content
        feat_mask = feat_mask.bool()  # make sure it's a boolean
        feat_mask = feat_mask.view(B, 1, -1).permute(0, 2, 1).squeeze(-1) # (B, H*W, 1)

        if len(self.training_tasks) != 0:
            task = self.training_tasks[0]

            # Context preparation
            memory, input_embs, mem_mask, input_masks, pad_label_ids= self._prepare_context_and_input(
                task=task,
                ctxt_ids=inputs[task]['ctxt_ids'],
                input_ids=inputs[task]['input_ids'],
                fw=feat_map,
                fw_masks=feat_mask,
                label_ids=inputs[task]["label_ids"]
            )

            ctxt_ids = inputs[task]['ctxt_ids'] if self.with_contexts[task] else None
            # Mode selection
            if self.is_autoregs[task]:
                
                pred_ids, pred_logits = self._generate_autoreg(
                    task=task,
                    memory=memory,
                    mem_mask=mem_mask,
                    max_len=max_len,
                    temperature=temperature,
                    ctxt_ids=ctxt_ids
                )
            else:
                pred_ids, pred_logits = self._generate_parallel(
                    task=task,
                    input_embs=input_embs,
                    memory=memory,
                    mem_mask=mem_mask,
                    max_len=max_len,
                    temperature=temperature,
                    ctxt_ids=ctxt_ids
                )
            outputs["output_ids"] = pred_ids
            outputs["logits"] = pred_logits

        return outputs

    def _generate_autoreg(
        self,
        task,
        memory,
        mem_mask,
        max_len,
        temperature,
        ctxt_ids=None,
    ):
        """
        Autoregressive generation with a fixed context.
        input = [context ; generated_tokens]
        The outputs contain ONLY the generated tokens.
        """

        device = memory.device
        B = memory.size(0)

        bos_id = self.tokenizers[task].bos_token_id
        eos_id = self.tokenizers[task].eos_token_id

        pred_ids = []
        pred_logits = []

        # ------------------------------------------------
        # 1. FIXED CONTEXT (pre-computed)
        # ------------------------------------------------
        if ctxt_ids is not None:
            ctxt_len = ctxt_ids.size(1)
            ctxt_embs, ctxt_mask, ctxt_last_pos = self._embed_and_mask(ctxt_ids, task)
            pos_offset = ctxt_last_pos  # position of the last context token
        else:
            ctxt_len = 0
            ctxt_embs, ctxt_mask = None, None
            pos_offset = 0

        # ------------------------------------------------
        # 2. INITIAL INPUT = <BOS>
        # ------------------------------------------------
        input_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

        # ------------------------------------------------
        # 3. AUTOREGRESSIVE LOOP
        # ------------------------------------------------
        for _ in range(max_len):
            # Embedding of all tokens generated so far
            inp_embs, _, _ = self._embed_and_mask(
                input_ids,
                task,
                pos_offset=pos_offset
            )

            # Concat context + generated tokens
            if ctxt_embs is not None:
                input_embs = torch.cat([ctxt_embs, inp_embs], dim=1)
                #input_masks = torch.cat([ctxt_mask, inp_mask], dim=1)
            else:
                input_embs = inp_embs

            # Decoder
            logits = self._run_decoder(
                task=task,
                input_embs=input_embs,
                input_masks=None,
                memory=memory,
                memory_key_padding_mask=mem_mask,
            )


            # Logits of the last generated token
            logits = logits[:, -1, :]
            pred_logits.append(logits)

            # Sampling
            if temperature == 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            pred_ids.append(next_id)

            # Stop if EOS everywhere
            if eos_id is not None and (next_id == eos_id).all():
                break

            # Append to the generated input
            input_ids = torch.cat([input_ids, next_id], dim=1)

        # ------------------------------------------------
        # 4. OUTPUTS (WITHOUT CONTEXT)
        # ------------------------------------------------
        pred_ids = torch.cat(pred_ids, dim=1) if pred_ids else None
        if ctxt_len > 0:
            # only keep the tokens generated after the context (BOS included here)
            pred_ids = pred_ids[0:]
        if pred_logits:
            # list[T] of [B,V]  --> [B,T,V]
            pred_logits = torch.stack(pred_logits, dim=1)
        return pred_ids, pred_logits

   
    def _generate_parallel(self, task, input_embs, memory, mem_mask, max_len, temperature, ctxt_ids=None):
        """
        Parallel (non-autoregressive) generation for a given task.
        """
        B = memory.size(0)

      
        logits = self._run_decoder(
                task=task,
                input_embs=input_embs,
                input_masks=None,
                memory=memory,
                memory_key_padding_mask=mem_mask,
            )
        
        if temperature == 0:
            pred_ids = torch.argmax(logits, dim=-1)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            pred_ids = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)
            pred_ids = pred_ids.view(B, max_len)

        return pred_ids, logits