import os
import random
import numpy as np
import json
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from torchvision.transforms.v2 import ElasticTransform, InterpolationMode

import matplotlib.pyplot as plt 
from math import ceil
from torchvision import transforms

import os
import random
from math import ceil
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from data.transforms import aug_config, apply_data_augmentation
from collections import deque

def fixed_context_orders(context_files, K):
    n = len(context_files)
    if n == 0:
        return [[]]  # no context
    if n == 1 or K == 1:
        return [context_files]  # trivial case: no permutation

    orders = []
    base = list(context_files)

    # 1) natural order
    orders.append(base)

    # 2) reversed order
    orders.append(list(reversed(base)))

    # 3+) rotations
    from collections import deque
    dq = deque(base)
    for _ in range(1, n):
        dq.rotate(1)
        orders.append(list(dq))

    # 4+) adjacent swaps
    for i in range(n - 1):
        s = base.copy()
        s[i], s[i + 1] = s[i + 1], s[i]
        orders.append(s)

    return orders[:K]



class FewShotWriterDataset(Dataset):
    def __init__(self, base_path, num_context_lines=10, use_image_aug=False, mode="train"):
        self.base_path = base_path
        self.num_eval_orders = 1
        self.eval_with_full_document = False #ATTN
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif'}
        self.writer_folders = [
            os.path.join(base_path, f)
            for f in sorted(os.listdir(base_path))
            if os.path.isdir(os.path.join(base_path, f))
            and sum(
                1 for file in sorted(os.listdir(os.path.join(base_path, f)))
                if os.path.splitext(file)[1].lower() in image_extensions
            ) >= 2
        ]
       

        self.writer_id_dict = None
        # 🔹 writer_folder → writer_id mapping
        self.folder_to_writer_id = {}
        if self.writer_id_dict is not None:
            for writer_id, file_ids in self.writer_id_dict.items():
                for folder in self.writer_folders:
                    folder_name = os.path.basename(folder)
                    if folder_name in file_ids:
                        self.folder_to_writer_id[folder] = writer_id

        self.use_image_aug=use_image_aug
        self.to_tensor = T.Compose([
            T.ToTensor(),
        ])
        
        self.num_context_lines = num_context_lines
        print(f"{len(self.writer_folders)} pages loaded from {self.base_path}")

        self.da_config = aug_config(0.9, 0.1)
        self.mode = mode 

        # ------------------------------------------------------------------
        # 🔁 Added: build the mapping so that every image serves as a query once
        # ------------------------------------------------------------------
        self.base_index_map = []  # list of (writer_idx, query_idx) tuples
        self.writer_images = []   # list of image files for each writer

        for w_idx, folder in enumerate(self.writer_folders):
            image_files = sorted([
                f for f in os.listdir(folder)
                if os.path.splitext(f)[1].lower() in image_extensions
            ])
            self.writer_images.append(image_files)

            # Each image of the writer serves as a query once
            for q_idx in range(len(image_files)):
                self.base_index_map.append((w_idx, q_idx))

        # ----------------------
        # Build index_map based on num_eval_orders
        # ----------------------
        self.index_map = []

        for base_idx in range(len(self.base_index_map)):
            for order_idx in range(self.num_eval_orders):
                self.index_map.append((base_idx, order_idx))

        self.total_samples = len(self.index_map)

        self.context_orders_map = {}
        for base_idx, (w_idx, q_idx) in enumerate(self.base_index_map):
            image_files = self.writer_images[w_idx]

            # Standard context: preceding lines
            if q_idx > 0:
                start = max(0, q_idx - self.num_context_lines)
                context_files = image_files[start:q_idx]
            else:
                context_files = [image_files[1]] if len(image_files) > 1 else []

            self.context_orders_map[base_idx] = fixed_context_orders(
                context_files,
                self.num_eval_orders
            )

        self.ocr_index_map = [x for x in self.base_index_map for _ in range(self.num_eval_orders)] 
        # ------------------------------------------------------------------
        if self.num_context_lines == 0:
            self._getitem_impl = self._get_target
        elif self.num_context_lines > 0:
            self._getitem_impl = self._get_target_and_context
       

    def __len__(self):

        # In validation/test, there is one entry per image (each image becomes a query once)
        if self.mode in ["generation", "val", "valid", "validation", "test"]:
            return len(self.index_map)

        # In training, keep the original behavior: one entry per writer
        return len(self.writer_folders)

    def __getitem__(self, idx):
        return self._getitem_impl(idx)

    def _add_small_rdn_padding(self, image):
        pad_left = random.randint(10, 20)
        pad_right = random.randint(5, 10)
        pad_top = random.randint(5, 10)
        pad_bottom = random.randint(5, 10)
        image = TF.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=255) #left, top, right and bottom
        return image
    
    def _resize_for_ctc(self, image, text=None, scale_factor=None, stride=8):
        """
        Two modes:
        - text provided       : computes the scale_factor required by the CTC constraint
        - scale_factor provided : directly applies the scale_factor (for context images)

        Returns: (resized_image, scale_factor)
        """
        w = image.width
        h = image.height

        if scale_factor is None:
            # Query mode: compute the required scale_factor
            text_len = len(text)  # without <start>/<end>
            min_feature_width = 2 * text_len
            min_pixel_width = min_feature_width * (8 + 4)

            if w < min_pixel_width:
                required_width = int(min_pixel_width)
                scale_factor = required_width / w
            else:
                scale_factor = 1.0  # no resize needed

        # Apply the scale_factor
        if scale_factor != 1.0:
            required_width  = int(w * scale_factor)
            required_height = int(h * scale_factor)

            image = TF.resize(
                image,
                (required_height, required_width),
                interpolation=InterpolationMode.BILINEAR
            )

        return image, scale_factor
    
    def _add_padding_for_ctc(self, image, text):
      # add left/right padding so that (feature_map_width) > text_length*2 -1 (condition required for CTC)
        text_len = len(text) + 4 #account for the <start> and <end> tokens
        min_width = (2 * text_len) # Minimum width required for CTC
        feature_map_width = image.width // 8  # assuming the CNN reduces the width by a factor of 8

        pad_left = random.randint(5, 20)
        pad_right = random.randint(0, 10)
        pad_top = random.randint(5, 10)
        pad_bottom = random.randint(5, 10)
        if feature_map_width < min_width:
            required_width = (min_width * 8)  # image width required for the feature map to be wide enough
            required_width = int(required_width * 1.50)  # add 50% margin to be safe
            padding_needed = required_width - image.width
            pad_right = padding_needed - pad_left
            image = TF.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=255) #left, top, right and bottom
        else:
            # add a bit of padding on the left
            image = TF.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=255) #left, top, right and bottom
        return image

    def _get_target_and_context(self, idx):
        if self.mode == "train":
            writer_id = None
            # Step 1: random selection of a writer
            writer_folder = random.choice(self.writer_folders)
            image_files = [f for f in os.listdir(writer_folder) if f.lower().endswith((".png", ".jpg", ".tif"))]

            if not image_files:
                raise ValueError(f"No image found in {writer_folder}")
            if len(image_files) < 2:
                raise ValueError(f"Only {len(image_files)} images found in {writer_folder}")
            # Step 2: random selection of a query
            query_file = random.choice(image_files)
            query_path = os.path.join(writer_folder, query_file)
            query_text_path = query_path.rsplit(".", 1)[0] + ".txt"

            with open(query_text_path, "r", encoding="utf-8") as f:
                query_text = f.read().strip()

            query_image = Image.open(query_path)  # .convert("L") if needed
            query_image, scale_factor = self._resize_for_ctc(query_image, query_text)
            query_img_original_widths = query_image.width
            query_image = self._add_small_rdn_padding(query_image)


            # Step 3: selection of context files (without loading unnecessary images)
            context_candidates = [f for f in image_files if f != query_file]
            max_context = min(len(context_candidates), self.num_context_lines)

            if self.eval_with_full_document:
                context_files = context_candidates
                context_files = random.sample(context_candidates, max_context)
            else :
                context_count = random.randint(1, max_context) 
                context_files = random.sample(context_candidates, context_count)


            # Load only the selected context images and texts
            context_samples = []
            for img_file in context_files:
                img_path = os.path.join(writer_folder, img_file)
                txt_path = img_path.rsplit(".", 1)[0] + ".txt"

                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                image = Image.open(img_path)  # .convert("L") if needed
                image, _ = self._resize_for_ctc(image, text, scale_factor=scale_factor)  # apply the same scale_factor as for the query
                image = self._add_small_rdn_padding(image)
                context_samples.append((image, text))


        else:
            # Index retrieval
            base_idx, order_idx = self.index_map[idx]
            writer_idx, query_idx = self.base_index_map[base_idx]

            writer_folder = self.writer_folders[writer_idx]
            image_files = self.writer_images[writer_idx]
            if self.folder_to_writer_id is not None:
                try:
                    writer_id = self.folder_to_writer_id[writer_folder]
                except KeyError:
                    writer_id = None
            else:
                writer_id = None  
            # --- Query ---
            query_file = image_files[query_idx]
            query_path = os.path.join(writer_folder, query_file)
            query_text_path = query_path.rsplit(".", 1)[0] + ".txt"

            with open(query_text_path, "r", encoding="utf-8") as f:
                query_text = f.read().strip()
            query_image = Image.open(query_path)
            query_image, scale_factor = self._resize_for_ctc(query_image, query_text)
            query_img_original_widths = query_image.width
            query_image = self._add_small_rdn_padding(query_image)

        
            context_candidates = [f for f in image_files if f != query_file]
            max_context = min(len(context_candidates), self.num_context_lines)

            if self.eval_with_full_document:
                context_files = context_candidates
                context_files = context_candidates[:max_context] #random.sample(context_candidates, max_context)
            else :
                context_count = random.randint(1, max_context) 
                context_files = random.sample(context_candidates, context_count)

        
            # --- Generation of fixed orders ---
            context_orders = fixed_context_orders(context_files, self.num_eval_orders)

            # --- Selection of the order corresponding to order_idx ---
            #context_files_ordered = context_orders[order_idx]
            # If the context is too short for K orders, loop over the available orders
            num_orders_available = len(context_orders)
            safe_order_idx = order_idx % num_orders_available
            context_files_ordered = context_orders[safe_order_idx]

            # --- Loading the context images and texts ---
            context_samples = []
            for img_file in context_files_ordered:
                img_path = os.path.join(writer_folder, img_file)
                txt_path = img_path.rsplit(".", 1)[0] + ".txt"

                # Option: replace with test_correct if it exists
                # txt_path_correct = Path(str(txt_path).replace("/test/", "/test_correct/"))
                # txt_path = txt_path_correct if txt_path_correct.exists() else txt_path

                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                image = Image.open(img_path)
                image, _ = self._resize_for_ctc(image, text, scale_factor=scale_factor)  # apply the same scale_factor as for the query
                image = self._add_small_rdn_padding(image)

                context_samples.append((image, text))

        context_texts = [txt for _, txt in context_samples]

        if self.use_image_aug:
            context_images = [
                apply_data_augmentation(img, self.da_config)
                for img, _ in context_samples
            ]
           
        else:
            context_images = [self.to_tensor(img) for img, _ in context_samples]
       
        # Find the max width among the context images
        context_max_width = max(img.shape[2] for img in context_images)

        # Pad each context image to that width
        def pad_width_to(img_tensor, width):
            _, h, w = img_tensor.shape
            if w < width:
                pad_right = width - w
                return TF.pad(img_tensor, (0, 0, pad_right, 0), fill=1.0) #left, top, right and bottom
            return img_tensor  # do nothing if already at the right width

        padded_context_images_masks = [torch.zeros((1, img.shape[-2], img.shape[-1]) , dtype=torch.uint8) for img in context_images ]  # mask for the context
        padded_context_images = [pad_width_to(img, context_max_width) for img in context_images]

        # Vertical concatenation of the context
        padded_context_images_masks = [pad_width_to(mask, context_max_width) for mask in padded_context_images_masks]
        padded_context_images_masks = torch.cat(padded_context_images_masks, dim=1)  # dim=1 = height
        context_concat = torch.cat(padded_context_images, dim=1)  # dim=1 = height

        # Load the query
        if self.use_image_aug:
            query_tensor = apply_data_augmentation(query_image, self.da_config)

        else:
            query_tensor = self.to_tensor(query_image)

        padded_query_images_mask = torch.zeros((1, query_tensor.shape[-2], query_tensor.shape[-1]), dtype=torch.uint8)  # mask for the query



        # Step 8: convert to 3 channels (RGB-like)
        context_rgb = context_concat.expand(3, -1, -1)
        query_rgb = query_tensor.expand(3, -1, -1)

        # Step 9: final stacking
        output_tensor = [context_rgb, query_rgb]  # torch.Size([2, 3, H, W])
        all_images_masks = [padded_context_images_masks, padded_query_images_mask]  # torch.Size([2, 1, H, W])
        # Concatenated text
        context_text = "ⓝ".join(context_texts)

        all_texts = [context_text, query_text]
        item_dict = {
            "scripter_ids" : writer_folder,
            "all_images" : output_tensor,
            "all_images_masks" : all_images_masks,
            "all_texts" : all_texts,
            "writer_id": writer_id,
            "query_file": query_file,
            "query_img_original_widths": query_img_original_widths,
        }
        return item_dict


    def _get_target(self, idx):

        if self.mode == "train":
            # Step 1: random selection of a writer
            writer_folder = random.choice(self.writer_folders)
            image_files = [f for f in os.listdir(writer_folder) if f.lower().endswith((".png", ".jpg", ".tif"))]

            # Step 2: random selection of a query
            query_file = random.choice(image_files)

        else :
            writer_idx, query_idx = self.ocr_index_map[idx]
            writer_folder = self.writer_folders[writer_idx]
            image_files = self.writer_images[writer_idx]
            # --- Query ---
            query_file = image_files[query_idx]

        # 🔹 ADDED: retrieve the writer-id
        if self.folder_to_writer_id is not None:
            writer_id = self.folder_to_writer_id[writer_folder]
        else:
            writer_id = None 
        
        query_path = os.path.join(writer_folder, query_file)
        query_text_path = query_path.rsplit(".", 1)[0] + ".txt"
        with open(query_text_path, "r", encoding="utf-8") as f:
            query_text = f.read().strip()
        query_image = Image.open(query_path)
        query_image, scale_factor = self._resize_for_ctc(query_image, query_text)
        query_img_original_widths = query_image.width
        query_image = self._add_small_rdn_padding(query_image)
      
        if self.use_image_aug:
            query_image = apply_data_augmentation(query_image, self.da_config)
        else:
            query_image = self.to_tensor(query_image)
        
        query_image_mask = torch.zeros((1, query_image.shape[-2], query_image.shape[-1]), dtype=torch.uint8)  # mask for the query
        query_image = query_image.expand(3, -1, -1) #to RBG

        item_dict = {
            #"scripter_ids" : writer_folder,
            "all_images" : [query_image],
            "all_texts" : [query_text],
            "all_images_masks" : [query_image_mask],
            "writer_id": writer_id,
            "query_img_original_widths": query_img_original_widths,
        }
        return item_dict

    