import os
import importlib
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
from jiwer import cer

from tokenizer.generic_tokenizer import GenericTokenizers
from model.dropout_scheduler import DropoutScheduler
from model.generic_model import GenericModel
from data.init_rosetta_data import init_dataloaders
from utils.utils import RosettaConfigDict, define_checkpoint_name, ctc_greedy_decode


class Trainer:
    def __init__(self, config, model, tokenizers, train_loader, valid_synth_loaders=None,
                 valid_real_loader=None, device="cuda", lr=1e-4, weight_decay=0.0,
                 save_path=None, pad_token_id=0):

        self.config = config
        self.training_tasks = config.model_config.training_tasks
        self.model_config = config.model_config
        self.model = model.to(device)
        self.tokenizers = tokenizers
        self.train_loader = train_loader
        self.valid_synth_loaders = valid_synth_loaders
        self.valid_real_loader = valid_real_loader

        self.device = device
        self.num_epochs = config.train_config.num_epochs
        self.save_path = save_path
        self.pad_token_id = pad_token_id

        self.dropout_scheduler = DropoutScheduler(encoder_config=config.model_config.encoder,
                                                  fcn_encoder=self.model.encoder)

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.best_valid_metric = float("inf")
        self.best_metric = self.config.train_config.best_metric
        self.current_p_synth = self.config.data_config.start_p_synth
        self.current_num_context_lines = self.config.data_config.start_num_context_lines
        self.max_num_context_lines = self.config.data_config.max_num_context_lines
        self.is_only_data_synth = config.data_config.end_p_synth ==1.0

        self.valid_cer_threashold = self.config.train_config.valid_cer_threashold

        self.target_H = None
        self.target_W = None

    def p_synth_update(self):
        """
        Dynamically updates the p_synth probability in the dataset.
        Allows resuming training at a given epoch without restarting the decay from 0.

        Args:
            self.epoch (int): current epoch (0-indexed)
            self.config.data_config.start_epoch_p_synth (int): epoch from which the decay starts
            self.config.data_config.end_epoch_p_synth (int): epoch at which the decay stops
        """
        start_p = self.config.data_config.start_p_synth
        end_p = self.config.data_config.end_p_synth
        start_epoch = getattr(self.config.data_config, "start_epoch_p_synth", 0)
        end_epoch = self.config.data_config.end_epoch_p_synth

        # Before the start: no decay yet
        if self.epoch <= start_epoch:
            new_p = start_p
        # After the end: final value reached
        elif self.epoch >= end_epoch:
            new_p = end_p
        else:
            # Normalized progress between start_epoch and end_epoch
            progress = (self.epoch - start_epoch) / (end_epoch - start_epoch)
            new_p = start_p + (end_p - start_p) * progress

        # Dataset update
        self.train_loader.dataset.set_p_synth(new_p)
        self.current_p_synth = new_p

    def train(self, expe_name, start_epoch=0):
        if self.config.train_config.use_wandb:
            wandb.init(project="Hybrid_Rosetta", name=expe_name)
            wandb.watch(self.model, log="all")

        for epoch in range(start_epoch, self.num_epochs + 1):
            self.epoch = epoch
            self.p_synth_update()
            self.dropout_scheduler.update_dropout_rate(self.epoch)

            #=== TRAIN PASS ===
            metrics = self.train_generic_epoch()
            if self.config.train_config.use_wandb:
                for key, value in metrics.items():
                    if len(value) > 0:
                        wandb.log({f"train {key}": np.mean(value)}, step=self.epoch)

                wandb.log({"synth ratio": self.current_p_synth}, step=self.epoch)
                wandb.log({"dropout rate": self.dropout_scheduler.current_dropout_rate}, step=self.epoch)
                wandb.log({"current_num_context_lines": self.current_num_context_lines}, step=self.epoch)

            # === VALID PASSES ===
            if (epoch + 1) % self.config.train_config.valid_interval == 0:
                for name, valid_synth_loader in self.valid_synth_loaders.items():
                    synth_metrics = self.evaluate_one_epoch(valid_synth_loader,
                                                            apply_ttt=False)
                    if self.config.train_config.use_wandb:
                        for key, value in synth_metrics.items():
                            if len(value) > 0:
                                wandb.log({f"val {name} {key}": np.mean(value)}, step=self.epoch)

                    if self.best_metric in synth_metrics.keys():
                        mean_best_metric = np.mean(synth_metrics[self.best_metric])

                # === REAL VALID PASS ===
                if self.valid_real_loader is not None:
                    real_metrics = self.evaluate_one_epoch(self.valid_real_loader,
                                                               apply_ttt=False)
                    if self.config.train_config.use_wandb:
                        for key, value in real_metrics.items():
                            if len(value) > 0:
                                wandb.log({f"valid real {key}": np.mean(value)}, step=self.epoch)


                    if self.best_metric in real_metrics.keys():
                        mean_best_metric = np.mean(real_metrics[self.best_metric])

                # === UPDATE the number of lines in the context ===
                if  mean_best_metric< self.valid_cer_threashold and self.current_num_context_lines < self.max_num_context_lines:
                    self.current_num_context_lines += 1
                    self.train_loader.dataset.real_dataset.num_context_lines +=1
                    self.train_loader.dataset.synth_dataset.num_context_lines +=1
                    try:
                        self.valid_real_loader.dataset.num_context_lines +=1
                        self.valid_synth_loader.dataset.num_context_lines +=1
                    except:
                        print("No real valid loader to update")
                    print(f"Update Current num context lines to : {self.current_num_context_lines} lines")

                # === SAVE the best model ===
                if mean_best_metric < self.best_valid_metric:
                    self.best_valid_metric = mean_best_metric
                    self.save_model(checkpoint_name='best_'+expe_name)
                    print(f"Save the best model with {self.best_metric}={mean_best_metric} at epoch {self.epoch}")
            self.save_model(expe_name)


    def evaluate(self, start_epoch=0):
        self.epoch = start_epoch

        self.p_synth_update()
        self.dropout_scheduler.update_dropout_rate(self.epoch)

        if self.valid_real_loader is not None:
            real_metrics = self.evaluate_test_one_epoch(self.valid_real_loader)
            for key, value in real_metrics.items():
                if len(value) > 0:
                    mean_val = round(np.mean(value) * 100,2)
                    median_val = round(np.median(value) * 100, 2)
                    print(f"valid IAM {key}: {mean_val}% (median = {median_val}%)")
                    plt.boxplot(value)
                    plt.title(f"Mean IAM = {mean_val}%")
                    plt.text(
                        x=1.1,
                        y=median_val / 100,
                        s=f"Median = {median_val}%",
                        color="orange",
                        va="center")
                    plt.show()

    def move_to_device(self, obj):
        """
        Recursively moves tensors (or PyTorch structures) to the specified device.
        """
        if torch.is_tensor(obj):
            return obj.to(self.device)
        elif isinstance(obj, dict):
            return {k: self.move_to_device(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.move_to_device(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self.move_to_device(v) for v in obj)
        else:
            return obj


    def plot_the_ith_image(self, images, index=0):
        img = images[index].cpu().permute(1, 2, 0).detach().numpy()
        plt.imshow(img)
        plt.show()


    def plot_batch(self,inputs):
        # Retrieve the batch
        batch_imgs = inputs["images"].cpu()  # [B, C, H, W]
        batch_size = batch_imgs.shape[0]

        # --- Full-screen figure creation ---
        plt.figure(figsize=(5, batch_size * 3))  # fixed width, proportional height
        manager = plt.get_current_fig_manager()
        try:
            manager.full_screen_toggle()
        except:
            pass

        # --- Vertical display ---
        for i in range(batch_size):
            img = batch_imgs[i].permute(1, 2, 0).numpy()
            plt.subplot(batch_size, 1, i + 1)
            plt.imshow(img)

            if isinstance(inputs["font_paths"][i], str):
                title = inputs["font_paths"][i].split("/")[-1]
            else:
                title = "real data"

            plt.title(title, fontsize=10)

        plt.tight_layout()
        plt.show()


    def train_generic_epoch(self):
        """
        Trains the model for one full epoch over the configured tasks.
        Handles encoding, the forward pass through the model, the loss, backpropagation,
        and the CER / TER metrics.
        """
        self.model.train()
        total_loss = 0.0
        cers, cers_ctc, ters = [], [], []

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}/{self.num_epochs}")

        if len(self.training_tasks) == 0:
            task = "cnn_ctc"  # By default, assume the task is OCR if no task is specified
        else:
            task = self.training_tasks[0]

        all_h, all_w = [], []
        for batch_idx, inputs in enumerate(self.train_loader):

            try:
                inputs = self.move_to_device(inputs)
                h, w = inputs["images"].shape[-2:]
                all_h.append(h)
                all_w.append(w)

                inputs["fw_2D"], inputs["fw_masks_2D"] = self.model.encoder(inputs["images"], inputs["images_masks"])
                inputs["stride"] = inputs["images"].shape[-1] / inputs["fw_2D"].shape[-1]  # compute the actual stride after the CNN
                del inputs["images"],
                del inputs["images_masks"]

                # === Forward pass through the model ===
                outputs = self.model(inputs)

                # === Loss computation === sums the losses if they are not None
                alpha = 0.5  # to be tuned
                loss_main = outputs.get("loss")
                loss_ctc = outputs.get("loss_ctc")

                if loss_main is not None and loss_ctc is not None:
                    loss = alpha * loss_main + (1 - alpha) * loss_ctc
                elif loss_main is not None:
                    loss = loss_main
                elif loss_ctc is not None:
                    loss = loss_ctc
                else:
                    raise ValueError("No loss available in outputs")
                #Gradient computation
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # === Logits decoding & metrics computation ===
                if self.model_config.is_cnn_ctc:
                    pred_ids_ctc = ctc_greedy_decode(logits=outputs["logits_ctc"],
                                        blank_id=self.tokenizers["ocr"].ctc_blank_token_id)

                    pred_strs_ctc = self.tokenizers["ocr"].decode_output_ids(
                            pred_ids_ctc, before="<end>"
                        )
                    labels_strs = inputs["ocr"]["label_strs"]
                    batch_cers = [cer(l, p) for l, p in zip(labels_strs, pred_strs_ctc)]
                    cers_ctc.extend(batch_cers)


                if task == "ocr":
                    if self.model.is_autoregs["ocr"]:
                        output_ids = torch.argmax(outputs["logits"], dim=-1).detach().cpu()
                        pred_strs = self.tokenizers[task].decode_output_ids(
                            output_ids, before="<end>"
                        )
                    else:
                        output_ids = ctc_greedy_decode(logits=outputs["logits"],
                                    blank_id=self.tokenizers[task].ctc_blank_token_id)

                        pred_strs = self.tokenizers[task].decode_output_ids(
                                output_ids, before="<end>"
                            )

                    batch_cers = [cer(l, p) for l, p in zip(inputs[task]["label_strs"], pred_strs)]
                    cers.extend(batch_cers)


                elif task == "icl":
                    if self.model.is_autoregs["icl"]:
                        output_ids = torch.argmax(outputs["logits"], dim=-1).detach().cpu()
                        #[pad_ctxt, label_ids, pad]
                        mask = (outputs["pad_label_ids"] != self.tokenizers[task].pad_token_id).cpu()

                        pred_strs = self.tokenizers[task].decode_output_ids(
                            output_ids,
                            inputs[task]["reference_dicts"],
                            mask=mask,
                        )
                    else:
                        pred_ids = ctc_greedy_decode(logits=outputs["logits"],
                                    blank_id=self.tokenizers[task].ctc_blank_token_id)
                        pred_ids = self.tokenizers[task].decode(pred_ids)
                        pred_strs = self.tokenizers[task].decode_custom(pred_ids, reference_dicts=inputs[task]["reference_dicts"])

                    batch_ters = [cer(l, p) for l, p in zip(inputs[task]["label_strs"], pred_strs)]
                    ters.extend(batch_ters)

                # === Loss tracking and display ===
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)

                postfix = [f"loss: {avg_loss:.4f}"]
                if ters:
                    postfix.append(f"ter: {np.mean(ters):.3f}")
                if cers:
                    postfix.append(f"cer: {np.mean(cers):.3f}")
                if cers_ctc:
                    postfix.append(f"cer_ctc: {np.mean(cers_ctc):.3f}")

                progress_bar.set_postfix_str(" | ".join(postfix))
                progress_bar.update()

            except Exception as e:
                print(f"Error at batch {batch_idx}: {e}")


        progress_bar.close()
        self.target_H = int(np.median(all_h))
        self.target_W = int(np.median(all_w))
        outputs = {
            "ters": ters,
            "cers": cers,
            "cers_ctc": cers_ctc,
        }

        return outputs

    def evaluate_one_epoch(self, val_loader, apply_ttt=False):
        """
        Evaluates the model over one full validation/test epoch.
        Computes losses and metrics (CER / TER) without updating the weights.
        """
        self.model.eval()
        cers,cers_ctc, ters = [], [], []
        all_preds = []
        all_labels = []
        all_preds_ctc = []
        all_labels_ctc = []
        progress_bar = tqdm(val_loader, desc=f"[Eval] Epoch {self.epoch}/{self.num_epochs}")

        if len(self.training_tasks) == 0:
            task = "cnn_ctc"  # By default, assume the task is OCR if no task is specified
        else:
            task = self.training_tasks[0]

        if self.target_H is None:
            self.compute_target_HW()

        for batch_idx, inputs in enumerate(val_loader):
                inputs = self.pad_images_to_train_condition(inputs)
                inputs = self.move_to_device(inputs)

                with torch.no_grad():
                    inputs["fw_2D"], inputs["fw_masks_2D"] = self.model.encoder(inputs["images"], inputs["images_masks"])
                    inputs["stride"] = inputs["images"].shape[-1] / inputs["fw_2D"].shape[-1]  # compute the actual stride after the CNN

                del inputs["images_masks"]

                with torch.no_grad():
                    outputs = self.model.generate(inputs)

                # --- Decoding & metrics computation ---
                if self.model_config.is_cnn_ctc:
                    pred_ids_ctc = ctc_greedy_decode(logits=outputs["logits_ctc"],
                                        blank_id=self.tokenizers["ocr"].ctc_blank_token_id)

                    pred_strs_ctc = self.tokenizers["ocr"].decode_output_ids(
                            pred_ids_ctc, before="<end>"
                        )
                    labels_strs_ctc = inputs["ocr"]["label_strs"]
                    batch_cers = [cer(l, p) for l, p in zip(labels_strs_ctc, pred_strs_ctc)]
                    cers_ctc.extend(batch_cers)
                    all_preds_ctc.extend(pred_strs_ctc)
                    all_labels_ctc.extend(labels_strs_ctc)

                if task == "ocr":
                    if self.model.is_autoregs["ocr"]:
                        output_ids = torch.argmax(outputs["logits"], dim=-1).detach().cpu()
                        pred_strs = self.tokenizers[task].decode_output_ids(
                            output_ids, before="<end>"
                        )
                    else:
                        output_ids = ctc_greedy_decode(logits=outputs["logits"],
                                    blank_id=self.tokenizers[task].ctc_blank_token_id)

                        pred_strs = self.tokenizers[task].decode_output_ids(
                                output_ids, before="<end>"
                            )

                    batch_cers = [cer(l, p) for l, p in zip(inputs[task]["label_strs"], pred_strs)]
                    cers.extend(batch_cers)
                    all_preds.extend(pred_strs)
                    all_labels.extend(inputs[task]["label_strs"])

                elif task == "icl":
                    if self.model.is_autoregs["icl"]:
                        output_ids = torch.argmax(outputs["logits"], dim=-1).detach().cpu()

                        pred_strs = self.tokenizers[task].decode_output_ids(
                            output_ids,
                            inputs[task]["reference_dicts"],
                            mask=None,
                        )
                    else:
                        pred_ids = ctc_greedy_decode(logits=outputs["logits"],
                                    blank_id=self.tokenizers[task].ctc_blank_token_id)
                        pred_ids = self.tokenizers[task].decode(pred_ids)
                        pred_strs = self.tokenizers[task].decode_custom(pred_ids, reference_dicts=inputs[task]["reference_dicts"])

                    all_preds.extend(pred_strs)
                    all_labels.extend(inputs[task]["label_strs"])
                    batch_ters = [cer(l, p) for l, p in zip(inputs[task]["label_strs"], pred_strs)]
                    ters.extend(batch_ters)

                # --- Progressive display ---
                postfix = []
                if ters:
                    postfix.append(f"ter: {np.mean(ters):.3f}")
                if cers:
                    postfix.append(f"cer: {np.mean(cers):.3f}")
                if cers_ctc:
                    postfix.append(f"cer_ctc: {np.mean(cers_ctc):.3f}")

                progress_bar.set_postfix_str(" | ".join(postfix))
                progress_bar.update()

        progress_bar.close()

        if self.model_config.is_cnn_ctc:
            predictions_ctc = ''.join(all_preds_ctc)
            labels_ctc = ''.join(all_labels_ctc)
            global_cer_ctc = cer(labels_ctc, predictions_ctc)
            print(f'global_cer_ctc : {global_cer_ctc}')
            global_cer_ctc = [global_cer_ctc]
        else:
            global_cer_ctc = []
        predictions = ''.join(all_preds)
        labels = ''.join(all_labels)
        if 'icl' in self.training_tasks:
            global_ter = cer(labels, predictions)
            print(f'global_ter : {global_ter}')
            global_ter = [global_ter]
        else:
            global_ter = []
        if 'ocr' in self.training_tasks:
            global_cer = cer(labels, predictions)
            print(f'global_cer : {global_cer}')
            global_cer = [global_cer]
        else:
            global_cer = []

        return {
            "ters": ters,
            "cers": cers,
            "global_ter" :  global_ter,
            "global_cer" : global_cer,
            "global_cer_ctc" : global_cer_ctc
        }

    def compute_alpha_ratio(self, label_str):
        """Computes the coverage ratio (1 - proportion of out-of-context '*' characters) of a label."""
        ignore_chars = set(r" ,;?.:/-_!()[]")

        # String length ignoring certain characters
        length = sum(1 for c in label_str if c not in ignore_chars)

        # Number of '*' in the string
        count_star = label_str.count('*')

        # Alpha computation
        ooc_ratio = count_star / length if length > 0 else 0

        return 1 - ooc_ratio


    def store_text_inputs(self, val_loader):
        """Serializes the positions and characters of the ICL labels of a dataloader, for offline analysis."""
        all_labels_pos = []
        all_labels_char = []
        print("Storing text inputs for analysis...")

        for batch_idx, inputs in enumerate(tqdm(val_loader)):
            if batch_idx > 3000:
                break
            labels = inputs["icl"]["label_ids"]
            labels_strs = inputs["icl"]["label_strs"]

            all_labels_char.append(labels_strs[0])

            labels_pos = self.tokenizers["icl"].decode(labels)
            # labels_pos is a list of strings of the form ["<t1> < > <t2> <!> ...<tk>", ...] for each sequence in the batch
            # each element is bracketed as "< >", e.g.: "<ooc> <t1> < > <t15> <t3> <t14> <t13> < > <t5> <t0> <t8> <t1> <t13>"
            # the <tk> positions are decoded into the int k, a space '< >' is worth -2, an out-of-context character
            # '<ooc>' is worth -1, and any unrecognized token is worth -3
            decoded_labels_pos = []
            for label in labels_pos:
                decoded_label = []
                for token in label.split():
                    if token.startswith("<t") and token.endswith(">"):
                        try:
                            pos = int(token[2:-1])  # extract the number between <t and >
                            decoded_label.append(pos)
                        except ValueError:
                            decoded_label.append(-3)  # unrecognized token
                    elif token == "<ooc>":
                        decoded_label.append(-1)
                    elif token == "< >":
                        decoded_label.append(-2)
                    else:
                        decoded_label.append(-3)  # unrecognized token
                decoded_labels_pos.append(decoded_label)

            all_labels_pos.append(decoded_labels_pos[0])
        real_dataset = self.config.data_config.real_dataset
        with open(f"data/cat_analysis/cat_data_random/{real_dataset}_labels_pos.json", "w", encoding="utf-8") as f:
            data_to_store = {
                "metadata": {
                    "real_dataset": real_dataset,
                    "num_sequences": len(all_labels_pos),
                },
                "sequences_pos": all_labels_pos,
                "sequences_char": all_labels_char
            }
            json.dump(data_to_store, f, ensure_ascii=False, indent=4)
        print(f"Labels stored in {real_dataset}_labels_pos.json")


    def evaluate_test_one_epoch(self, val_loader):
        """
        Evaluates the model over one full test epoch.
        Computes the metrics (CER / TER) globally, per writer, and per number of context lines,
        without updating the weights.
        """
        self.model.eval()
        cers, ters, cers_ctc = [], [], []
        all_preds = defaultdict(list)
        all_labels = defaultdict(list)
        all_labels_per_writer_id = defaultdict(list)
        all_preds_per_writer_id = defaultdict(list)
        ter_per_line = defaultdict(list)

        ter_per_writer_id = defaultdict(list)
        cer_per_writer_id = defaultdict(list)
        alpha_per_line = defaultdict(list)

        progress_bar = tqdm(val_loader, desc=f"[Eval] Epoch {self.epoch}/{self.num_epochs}")

        idx_with_no_ooc_path = "idx_with_no_ooc_TEST.json"
        if os.path.exists(idx_with_no_ooc_path):
            with open(idx_with_no_ooc_path, "r", encoding="utf-8") as file:
                idx_with_no_ooc = json.load(file)
        else:
            idx_with_no_ooc = []

        if len(self.training_tasks) == 0:
            task = "cnn_ctc"  # By default, assume the task is OCR if no task is specified
        else:
            task = self.training_tasks[0]

        if self.target_H is None:
            self.compute_target_HW()

        for batch_idx, inputs in enumerate(val_loader):
            try:
                inputs = self.pad_images_to_train_condition(inputs)
                inputs = self.move_to_device(inputs)

                with torch.no_grad():
                    inputs["fw_2D"], inputs["fw_masks_2D"] = self.model.encoder(inputs["images"], inputs["images_masks"])
                    inputs["stride"] = inputs["images"].shape[-1] / inputs["fw_2D"].shape[-1]  # compute the actual stride after the CNN

                del inputs["images_masks"]

                with torch.no_grad():
                    outputs = self.model.generate(inputs)

                # --- Decoding & metrics computation ---
                if self.model_config.is_cnn_ctc:
                    pred_ids_ctc = ctc_greedy_decode(logits=outputs["logits_ctc"],
                                        blank_id=self.tokenizers["ocr"].ctc_blank_token_id)

                    pred_strs_ctc = self.tokenizers["ocr"].decode_output_ids(
                            pred_ids_ctc, before="<end>"
                        )
                    labels_strs_ctc = inputs["ocr"]["label_strs"]
                    batch_cers = [cer(l, p) for l, p in zip(labels_strs_ctc, pred_strs_ctc)]
                    cers_ctc.extend(batch_cers)

                if task in ["ocr", "icl"]:
                    output_ids = outputs["output_ids"].detach().cpu()

                    if task == "ocr":

                        if self.model.is_autoregs["ocr"]:
                            output_ids = torch.argmax(outputs["logits"], dim=-1).detach().cpu()
                            pred_strs = self.tokenizers[task].decode_output_ids(
                                output_ids, before="<end>"
                            )
                        else:
                            output_ids = ctc_greedy_decode(logits=outputs["logits"],
                                        blank_id=self.tokenizers[task].ctc_blank_token_id)

                            pred_strs = self.tokenizers[task].decode_output_ids(
                                    output_ids, before="<end>"
                                )

                        batch_cers = [cer(l, p) for l, p in zip(inputs[task]["label_strs"], pred_strs)]
                        cers.extend(batch_cers)

                        all_preds["cer"].extend(pred_strs)
                        all_labels["cer"].extend(inputs[task]["label_strs"])
                        all_preds_per_writer_id[str(inputs['writer_id'][0])].extend(pred_strs)
                        all_labels_per_writer_id[str(inputs['writer_id'][0])].extend(inputs[task]["label_strs"])
                        cer_per_writer_id[str(inputs['writer_id'][0])].extend(batch_cers)

                    if task == "icl":
                        if self.model.is_autoregs["icl"]:
                            output_ids = torch.argmax(outputs["logits"], dim=-1).detach().cpu()

                            pred_strs = self.tokenizers[task].decode_output_ids(
                                output_ids,
                                inputs[task]["reference_dicts"],
                                mask=None,
                            )
                        else:
                            pred_ids = ctc_greedy_decode(logits=outputs["logits"],
                                        blank_id=self.tokenizers[task].ctc_blank_token_id)
                            pred_ids = self.tokenizers[task].decode(pred_ids)
                            pred_strs = self.tokenizers[task].decode_custom(pred_ids, reference_dicts=inputs[task]["reference_dicts"])

                        num_line = (inputs[task]["ctxt_ids"] == 4).sum().item() + 1
                        labels = inputs["icl"]["label_strs"]

                        alpha = self.compute_alpha_ratio(labels[0])
                        alpha_per_line[str(num_line)].append(alpha)

                        batch_ters = [cer(l, p) for l, p in zip(inputs[task]["label_strs"], pred_strs)]
                        batch_ters = [c for c in batch_ters if c<=1.0]
                        ters.extend(batch_ters)
                        all_preds[str(num_line)].extend(pred_strs)
                        all_labels[str(num_line)].extend(inputs[task]["label_strs"])
                        ter_per_line[str(num_line)].extend(batch_ters)

                        try :
                            all_preds_per_writer_id[str(inputs['writer_id'][0])].extend(pred_strs)
                            all_labels_per_writer_id[str(inputs['writer_id'][0])].extend(inputs[task]["label_strs"])

                            ter_per_writer_id[str(inputs['writer_id'][0])].extend(batch_ters)
                        except Exception as e:
                            # if one of the labels in the batch is an empty string ''
                            print(e)

                # --- Progressive display ---
                postfix = []
                if ters:
                    postfix.append(f"ter: {np.mean(ters):.3f}")
                if cers:
                    postfix.append(f"cer: {np.mean(cers):.3f}")
                if cers_ctc:
                    postfix.append(f"cer_ctc: {np.mean(cers_ctc):.3f}")
                progress_bar.set_postfix_str(" | ".join(postfix))
                progress_bar.update()

            except Exception as e:
                print(f"Error at batch {batch_idx}: {e}")
                print(inputs)
        progress_bar.close()

        if task == "icl":
            cers_ctc = ters
        else:
            cers_ctc = cers if cers_ctc == [] else cers_ctc

        # Distribution of the CER/TER on the test set
        plt.figure(figsize=(10, 5))
        cers_ctc = [c * 100 for c in cers_ctc]  # as a percentage
        median_cer_ctc = np.median(cers_ctc)
        print(f"Median CER (CTC) on test set: {median_cer_ctc:.2f}%")
        plt.boxplot(cers_ctc, showfliers=True)
        plt.title("CER (CTC) distribution on the test set")
        plt.ylabel("Char Error Rate (CER)")
        plt.xlabel("CTC Predictions")
        plt.show()

        print("store the list idx_with_no_ooc")
        with open(idx_with_no_ooc_path, "w", encoding="utf-8") as f:
            json.dump(idx_with_no_ooc, f, indent=2)

        if task == "icl":
            ter_global_per_line = {}
            all_predictions_flat = []
            all_labels_flat = []
            # Global TER computation per number of context lines
            for num_line in sorted(all_preds.keys(), key=lambda x: int(x)):
                predictions = ''.join(all_preds[num_line])
                labels = ''.join(all_labels[num_line])

                global_cer = cer(labels, predictions)
                ter_global_per_line[int(num_line)] = global_cer

                print(f'global_ter {num_line} line(s): {global_cer}')
                all_predictions_flat.append(predictions)
                all_labels_flat.append(labels)

            # Global TER regardless of the number of lines
            global_predictions = ''.join(all_predictions_flat)
            global_labels = ''.join(all_labels_flat)

            ter_global = cer(global_labels, global_predictions)
            print(f'\nGLOBAL TER (all lines combined): {ter_global}')

        return {
            "ters": ters,
            "cers": cers,
        }


    def compute_target_HW(self):
        """Computes the median dimensions (H, W) of the train set images, used for padding during evaluation."""
        print("Compute target_HW for padding")
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}/{self.num_epochs}")

        all_h, all_w = [], []
        for batch_idx, inputs in enumerate(self.train_loader):
            h, w = inputs["images"].shape[-2:]
            all_h.append(h)
            all_w.append(w)
            progress_bar.update()
        progress_bar.close()

        self.target_H = int(np.median(all_h))
        self.target_W = int(np.median(all_w))
        print(f"Target_H: {self.target_H}, Target_W: {self.target_W}")

    def pad_images_to_train_condition(self, inputs, padding_value=1.0):
        B, C, H, W = inputs["images"].shape

        pad_h = self.target_H - H
        pad_w = self.target_W - W

        # Only pad if necessary (positive values)
        if pad_h > 0 or pad_w > 0:
            pad_h = max(0, pad_h)
            pad_w = max(0, pad_w)

            # Image padding (right + bottom only)
            inputs["images"] = F.pad(
                inputs["images"],
                (0, pad_w, 0, pad_h),
                value=padding_value
            )

            # Mask padding (0 = padding)
            inputs["images_masks"] = F.pad(
                inputs["images_masks"],
                (0, pad_w, 0, pad_h),
                value=0
            )
            return inputs
        else:
            return inputs


    def load_encoder_model(self, load_path):
        """
        Loads only the model's encoder from load_path.
        Missing layers are randomly initialized.
        """

        # Load the checkpoint
        encoder_state_dict = torch.load(load_path, map_location='cpu', weights_only=False)['encoder_state_dict']

        # Partially load the weights
        missing, unexpected = self.model.encoder.load_state_dict(encoder_state_dict, strict=False)

        print(f"✅ Encoder loaded from: {load_path}")
        if missing:
            print(f"ℹ️ Parameters missing from the file for the encoder: {missing}")
        if unexpected:
            print(f"⚠️ Unexpected parameters ignored: {unexpected}")

        # Randomly reinitialize the missing layers
        if missing:
            print("🔄 Randomly reinitializing missing layers...")
            for name, module in self.model.encoder.named_modules():
                for param_name, param in module.named_parameters(recurse=False):
                    full_name = f"{name}.{param_name}" if name else f"{param_name}"
                    if full_name in missing:
                        if "weight" in param_name:
                            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                            else:
                                nn.init.normal_(param, mean=0.0, std=0.02)
                        elif "bias" in param_name:
                            nn.init.zeros_(param)

        print("✅ Missing layers randomly initialized.")

    def save_model(self, checkpoint_name):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'best_valid_metric' : self.best_valid_metric,
            'current_num_context_lines' : self.current_num_context_lines,
            'target_H': self.target_H,
            'target_W': self.target_W,
        }
        checkpoint_path = f"checkpoints/{checkpoint_name}.pt"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_model(self, load_path):
        if load_path is None or not os.path.exists(load_path):
            print(f"Starting training from scratch.")
            return 0  # Start from epoch 0

        print(f"📦 Loading model from {os.path.basename(load_path)}")
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)

        model_dict = self.model.state_dict()
        compatible_state_dict = {}
        skipped = []
        for k, v in checkpoint['model_state_dict'].items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    compatible_state_dict[k] = v
                else:
                    skipped.append((k, v.shape, model_dict[k].shape))
            else:
                skipped.append((k, v.shape, None))

        # Load only the compatible layers
        self.model.load_state_dict(compatible_state_dict, strict=False)


        print(f"✅ Partial weights loaded from {load_path}")
        if skipped:
            print("⚠️ Skipped layers (missing or different sizes):")
            for name, shape_ckpt, shape_model in skipped:
                print(f"  - {name}: {shape_ckpt} → {shape_model}")

        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("✅ Optimizer loaded from the checkpoint.")
        except Exception as e:
            print(f"⚠️ Error while loading the optimizer: {e}")

        if checkpoint['current_num_context_lines'] is  None:
            self.current_num_context_lines = 1
        else:
            self.current_num_context_lines = checkpoint['current_num_context_lines']
            if self.current_num_context_lines > self.config.data_config.max_num_context_lines:
                self.current_num_context_lines = self.config.data_config.max_num_context_lines

            # Update the max number of lines generated by the dataloaders
            try:
                # if mixed
                self.train_loader.dataset.real_dataset.num_context_lines = self.current_num_context_lines
                self.train_loader.dataset.synth_dataset.num_context_lines  = self.current_num_context_lines
            except:
                # if not mixed
                self.train_loader.dataset.num_context_lines = self.current_num_context_lines

            try:
                self.valid_real_loader.dataset.num_context_lines = self.current_num_context_lines
            except:
                print("No real valid loader to update")

            print(f"Update Current num context lines to : {self.current_num_context_lines} lines")

        if 'target_H' in checkpoint.keys():
            self.target_H = checkpoint['target_H']
            self.target_W = checkpoint['target_W']

        return checkpoint['epoch'] + 1


def main():
    parser = argparse.ArgumentParser(description="Training script with config")
    parser.add_argument('--config', type=str, required=True, help="Name of the config file without the '.py'")
    args = parser.parse_args()

    # --- Loading the configuration (model, data, training) ---
    config_path = f"config.{args.config}"
    config_module = importlib.import_module(config_path)
    params = config_module.params
    print(f"Configuration loaded from {args.config}:")
    np.random.seed(44)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = RosettaConfigDict(params)

    # --- Initialization of the tokenizers, model, dataloaders ---
    tokenizers = GenericTokenizers(config=config)
    model = GenericModel(model_config=config.model_config,
                         tokenizers=tokenizers,
                       ).to(device)
    train_loader, valid_real_loader, val_synth_loaders = init_dataloaders(config=config, tokenizers=tokenizers)
    # --- Initialization of the trainer that manages model training ---
    trainer = Trainer(config=config,
                      model=model,
                      tokenizers=tokenizers,
                      train_loader=train_loader,
                      valid_synth_loaders=val_synth_loaders,
                      valid_real_loader=valid_real_loader,
                      device=device,
                      lr=1e-4,
                      weight_decay=1e-4,
                      save_path="checkpoints",
                      pad_token_id=0)

    # --- Loading the model weights ---
    start_epoch = trainer.load_model(load_path=config.train_config.load_checkpoint_path)
    expe_name = define_checkpoint_name(config=config)
    print(f'Experiment: {expe_name}')

    # # --- Launching training  ---
    # trainer.train(expe_name=expe_name, start_epoch=start_epoch)

    # --- Launching evaluation  ---
    trainer.evaluate(start_epoch=start_epoch)


if __name__ =="__main__":
    main()
