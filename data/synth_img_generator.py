import json
import os
import random
import textwrap
import re
from math import ceil
from random import randint
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
from sympy import content
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.distributions.uniform import Uniform
from torch.utils.data import Dataset
from data.transforms import aug_config, apply_data_augmentation
import string

class SynthImageTextDataset(Dataset):
    def __init__(self, padding_value, context_content, font_paths, imgs_per_epoch, font_folder=None, target_content=[], image_augmentation=False, charset=None, num_context_lines=2):
        self.num_context_lines = num_context_lines
        self.padding_value = padding_value  # Fill value used for padding (-1 by default)

        # target_content is a list of dataset names (e.g. IAM, RIMES, CVL) or content types (e.g. "real_text", "random_letters", "wikipedia") used to generate the target text. Same for context_content, but for the context.
        self.target_contents = target_content if isinstance(target_content, list) else [target_content]
        self.context_contents = context_content if isinstance(context_content, list) else [context_content]
        
        self.font_folder = font_folder
        self.font_paths = font_paths

        # Load the font size dictionary from the JSON file
        font_dict_path = f"{font_folder}_fontsize_dict.json"

        if os.path.exists(font_dict_path):
            with open(font_dict_path, "r", encoding="utf-8") as file:
                self.font_dicts = json.load(file)
        else:
            print(f"⚠️  Dictionary file not found: {font_dict_path}")
            self.font_dicts = {}
  
        if any(content in ['RIMES', 'UKR', 'IAM', 'CVL', 'BNF'] for content in target_content):
            self.real_text_dataset = RealTextDataset(dataset_names=target_content)
      
        self.charset = charset
        self.imgs_per_epoch = imgs_per_epoch
        self.target_dicts = None
        self.current_font_idx = 0
        self.use_image_augmentation = image_augmentation
        self.da_config = aug_config(0.9, 0.1)
        
        if self.num_context_lines > 0:
            self._getitem_impl = self._get_context_and_target
        else:
            self._getitem_impl = self._get_target

    
    def __len__(self):
        return self.imgs_per_epoch

    def _load_json(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data

    def __getitem__(self, idx):
        return self._getitem_impl(idx)

    def random_text(self, n=20):
            alphabet = string.ascii_lowercase
            return "".join(random.choice(alphabet) for _ in range(n))


    def blank_image(self, H=80, W=500):
        return np.ones((H, W, 3), dtype=np.uint8) * 255
    
  

    def _get_context_and_target(self, idx):
        all_images = []
        all_images_masks = []
        all_texts = []
        
        # random choice of a content for the target:

        current_target_content = random.choice(self.target_contents)
        target_text = self.text_generator(current_target_content)
        font_dict = self.update_font_dict() # Pick a random font
        font_path = f"{self.font_folder}/{font_dict['font_name']}"
        font_size_target = font_dict["font_size"]


        if self.num_context_lines > 0 :
            context_text = ''
            # pick between 1 and num_context_lines
            current_num_context_lines = randint(1, self.num_context_lines)
            for idx in range(current_num_context_lines):
                context_text += self.text_generator(target_text=target_text,
                                                content=current_target_content,
                                                )
                if idx+1 != current_num_context_lines:
                    context_text += 'ⓝ'


            all_texts.append(context_text)
        all_texts.append(target_text)


        font_sizes_context = [font_size_target for _ in range(current_num_context_lines)]
       
        if self.use_image_augmentation:
            font_sizes_context = [fs + randint(-2, 2) for fs in font_sizes_context]
        
        context_imgs = []
        target_img = None
        # ===== EARLY FALLBACK =====
        if context_text.strip() == "":
            print("⚠ EMPTY CONTEXT TEXT -> fallback sample")

            n_ctx = max(1, len(font_sizes_context))

            context_text = "ⓝ".join(self.random_text() for _ in range(n_ctx))
            target_text = self.random_text()

            target_img = self.blank_image()
            context_imgs = [self.blank_image() for _ in range(n_ctx)]

        else:
        # ===== NORMAL PATH =====
            try :
                target_img = self.image_generator(target_text,
                                                font_path,
                                                font_size_target)
                
               

                target_img, scale_factor = self._resize_for_ctc(target_img, target_text)
                query_img_original_widths = target_img.shape[1]

                context_imgs = []
                while len(context_imgs) == 0: 
                    for text, font_size in zip(context_text.split('ⓝ'),font_sizes_context):
                        img = self.image_generator(text,
                                                    font_path,
                                                    font_size,                                                    
                                                    )
                        
                        img, _ = self._resize_for_ctc(img, scale_factor=scale_factor)
                        context_imgs.append(img)

                     
            except Exception as e: 
                print("⚠ IMAGE GENERATION FAILED")
                if target_img is None:
                    target_img = self.blank_image()
                if len(context_imgs) == 0:
                    n_ctx = len(context_text.split('ⓝ'))
                    context_imgs = [self.blank_image() for _ in range(n_ctx)]
        
        # ===== SAFETY =====
        if len(context_imgs) == 0:
            context_imgs = [self.blank_image()]

        if self.use_image_augmentation:
            context_imgs = [Image.fromarray(img) for img in context_imgs]
            context_imgs = [apply_data_augmentation(img, self.da_config,fill_value=255) for img in context_imgs]
        else:
            context_imgs = [torch.tensor(img).float() for img in context_imgs]
            context_imgs = [img.permute(2, 0, 1) for img in context_imgs]
            context_imgs = [img/255.0 for img in context_imgs]

        context_imgs_masks = [
            torch.zeros((1, img.shape[-2], img.shape[-1]), dtype=torch.uint8, device=img.device)
            for img in context_imgs
        ]

        max_height = max([img.shape[1] for img in context_imgs])
        max_context_width = max([img.shape[2] for img in context_imgs])
        max_width = max(max_context_width, target_img.shape[1])
        
        context_imgs = [self._pad_image(img, max_height, max_width) for img in context_imgs]
        context_imgs_masks = [self._pad_image(mask, max_height, max_width) for mask in context_imgs_masks]
        context_img = torch.concat(context_imgs, dim=1)
        context_imgs_masks = torch.concat(context_imgs_masks, dim=1)
        
        all_images.append(context_img)
        all_images_masks.append(context_imgs_masks)

        #add the target at the end
        if self.use_image_augmentation:
            target_img = Image.fromarray(target_img)
            target_img = apply_data_augmentation(target_img, self.da_config, fill_value=255)
        else:
            target_img = torch.tensor(target_img).float()
            target_img = target_img.permute(2, 0, 1)

            target_img = target_img / 255.0


        target_img_mask = torch.zeros((1, target_img.shape[-2], target_img.shape[-1]), dtype=torch.uint8, device=target_img.device)
        target_img = self._pad_image(target_img, target_img.shape[1], max_width)
        target_img_mask = self._pad_image(target_img_mask, target_img_mask.shape[1], max_width)
        all_images.append(target_img)
        all_images_masks.append(target_img_mask)

        #remove repeated spaces and leading/trailing spaces
        all_texts = [re.sub(r'\s+', ' ', text).strip() for text in all_texts]
        if query_img_original_widths is None:
            query_img_original_widths = target_img.shape[2]
            if query_img_original_widths is None:
                query_img_original_widths = max_width

        item_dict = {
            "font_path" : font_path,
            "font_sizes" : font_size_target,
            "all_images" : all_images,
            "all_images_masks" : all_images_masks,
            "all_texts" : all_texts,
            "query_img_original_widths": query_img_original_widths,
        }

        return item_dict


    def _get_target(self, idx):
        all_texts = []

        current_target_content = random.choice(self.target_contents)
        target_text = self.text_generator(current_target_content)
        font_dict = self.update_font_dict() # Pick a random font
        font_path = f"{self.font_folder}/{font_dict['font_name']}"
        font_size_target = font_dict["font_size"]

        all_texts.append(target_text)
        try :
            target_img = self.image_generator(target_text,
                                            font_path,
                                            font_size_target                                           ,
                                            )
            target_img, scale_factor = self._resize_for_ctc(target_img, target_text)
            query_img_original_widths = target_img.shape[1]
            target_img = self._add_small_rdn_padding(target_img)
            
        except Exception as e: 
            print(font_path)
            print(target_text)

        #add the target at the end
        if self.use_image_augmentation:
            target_img = Image.fromarray(target_img)
            target_img = apply_data_augmentation(target_img, self.da_config, fill_value=255)
        else:
            target_img = torch.tensor(target_img).float()
            target_img = target_img.permute(2, 0, 1)
            target_img = target_img / 255.0

        target_image_mask = torch.zeros((1, target_img.shape[-2], target_img.shape[-1]), dtype=torch.uint8, device=target_img.device)

        all_texts = [re.sub(r'\s+', ' ', text).strip() for text in all_texts]
        item_dict = {
            "font_path" : font_path,
            "font_sizes" : font_size_target,
            "all_images" : [target_img],
            "all_images_masks" : [target_image_mask],
            "all_texts" : all_texts,
            "query_img_original_widths": query_img_original_widths,
            }
        return item_dict

   
    def _pad_image(self, img, max_height, max_width):
        """
        Pads an image so that it has size (max_height, max_width).
        Padding is added on the right and bottom with a value of -1.
        :param img: Image as a tensor [C, H, W].
        :param max_height: Maximum height of the batch.
        :param max_width: Maximum width of the batch.
        :return: Padded image as a tensor [C, max_height, max_width].
        """
        _, img_height, img_width = img.shape
        pad_w = max_width - img_width
        pad_h = max_height - img_height

        # Apply padding only on the right and bottom (format: (left, right, top, bottom))
        padded_img = F.pad(img, (0, pad_w, 0, pad_h), value=self.padding_value)

        return padded_img


    def update_font_dict(self):
        """
        Selects a (font_name, size) pair depending on the mode:
        chooses a font at random
        """
        if not self.font_dicts:
            print("⚠️ No font dictionary loaded.")
            return None

        font_names = list(self.font_dicts.keys())
        font_name = random.choice(font_names)
        font_size = self.font_dicts.get(font_name, None)
    
        return {"font_name": font_name, "font_size": font_size}


    def update_font_path(self):
        font_path = random.choice(self.font_paths)
        return font_path
    
    
    def text_generator(self, content, target_text=None):
        """
        Generates a text based on the given content.
        :param content: Type of content to generate.
        :param target_text: text of the target image.
        :return: Generated text.
        """
        if content == 'random_letters':
            text = ''
            if len(text.strip()) == 0 :
                if self.max_num_tgt_letters == 1:
                    text += random.choice(self.charset)
                else:
                    while len(text.strip()) == 0: # ??
                        for _ in range(randint(1, self.max_num_tgt_letters)):
                            text += random.choice(self.charset)
            text = list(text)
            random.shuffle(text)
            text = ''.join(text)
        
        elif content == 'shuffle_target':
            text = list(target_text)
            random.shuffle(text)
            text = ''.join(text)

        elif content == 'alphabet':
            text = list(self.charset[self.num_context_char])

        
        elif content in ['RIMES', 'UKR', 'IAM', 'CVL', 'BNF']:
            text = self.real_text_dataset.get_real_text(dataset_name=content)   
        else:
            raise ValueError("Wrong content category")

        return text


    # @profile
    def _add_small_rdn_padding(self, image):
        #adds random padding on the left and right
        pad_left = random.randint(5, 10)
        pad_right = random.randint(5, 10)
        pad_top = random.randint(5, 10)
        pad_bottom = random.randint(5, 10)
        if image.ndim == 3:
            image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), constant_values=255)
        else:
            image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=255)
        return image

    def _resize_for_ctc(self, image, text=None, scale_factor=None, stride=8):
        """
        Two modes:
        - text provided      : computes the scale_factor required by the CTC constraint
        - scale_factor provided : directly applies the scale_factor (for context images)

        Returns: (resized_image, scale_factor)
        """
        h, w = image.shape[:2]

        if scale_factor is None:
            # Query mode: compute the required scale_factor
            text_len = len(text)  # without <start>/<end>
            min_feature_width = 2 * text_len
            min_pixel_width = min_feature_width * (8 + 4)

            if w < min_pixel_width:
                required_width = int(min_pixel_width)
                scale_factor = required_width / w  # e.g.: 1.8, 2.3...
            else:
                scale_factor = 1.0  # no resize needed

        # Apply the scale_factor
        if scale_factor != 1.0:
            required_width  = int(w * scale_factor)
            required_height = int(h * scale_factor)

            image = np.array(Image.fromarray(image).resize(
                (required_width, required_height),
                resample=Image.BICUBIC
            ))

        return image, scale_factor
    
    def _add_padding_for_ctc(self, image, text):
        # add left/right padding so that (feature_map_width) > text_length*2 -1 (condition required for CTC)
        h, w = image.shape[:2]
        text_len = len(text) + 4 #account for the <start> and <end> tokens
        min_feature_width = 2 * text_len
        downsample_factor = 8

        pad_left = random.randint(5, 20)
        pad_right = random.randint(0, 10)
        pad_top = random.randint(5, 10)
        pad_bottom = random.randint(5, 10)
    
        feature_map_width = w // downsample_factor
        if feature_map_width < min_feature_width:
            required_width = (min_feature_width * 8)  # image width required for the feature map to be wide enough
            required_width = int(required_width * 1.50)  # add 50% margin to be safe
            padding_needed = required_width - w
            pad_right = padding_needed - pad_left
            if image.ndim == 3:
                image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), constant_values=255)
            else:
                image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=255)
        else:
        # add a bit of padding on the left
            if image.ndim == 3:
                image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), constant_values=255)
            else:
                image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=255)
        return image

    def image_generator(self, text, font_path, font_size=randint(39, 40)):
        # font = ImageFont.truetype(font_path, size=font_size)
        font = get_cached_font(font_path, font_size)
        _, _, text_width, text_height = font.getbbox(text)
     
        # Vectorized padding computation
        ratios = rand_uniform_vec(
            [0.05, 0.05, 0.05, 0.05], #min
            [0.2, 0.2, 0.07, 0.07] #max
        )
        ratios_np = ratios.numpy()  # Convert to NumPy if needed
        padding_top, padding_bottom = (ratios_np[:2] * text_height).astype(int)
        padding_left, padding_right = (ratios_np[2:] * text_width).astype(int)
        padding = [padding_top, padding_bottom, padding_left, padding_right]
        if text == '':
            text = ' '
            padding = [5,5,5,5]
       
        img = self._generate_typed_text_line(text, font, padding, text_width=text_width, text_height=text_height)
            

        return img


    def white_image_generator(self, font_size=randint(39, 40), font_colors=None, add_box=False):
        font = ImageFont.load_default(size=font_size)
        text = ' '
        padding = [5,5,5,5]
        img = self._generate_typed_text_line(text, font, padding, font_colors=font_colors, add_box=add_box)

        return img

    def _generate_typed_text_line(self, text, font, padding, color_mode="RGB",text_width=None, text_height=None):
        padding_top, padding_bottom, padding_left, padding_right = padding
        if text_width is None or text_height is None:
            _, _, text_width, text_height = font.getbbox(text)
        img_height = padding_top + padding_bottom + text_height
        img_width = padding_left + padding_right + text_width
        img = Image.new(color_mode, (img_width, img_height), color=(255,255,255))
        d = ImageDraw.Draw(img)

        d.text((padding_left, padding_bottom), text, font=font, fill=(0,0,0), spacing=0)
        return np.array(img)


def rand_uniform_vec(lows, highs):
    """
    Vectorized version of rand_uniform using PyTorch.
    Preserves randomness across DataLoader workers.
    """
    lows = torch.tensor(lows, dtype=torch.float32)
    highs = torch.tensor(highs, dtype=torch.float32)
    return Uniform(lows, highs).sample()


def collate_fn(batch, with_context, padding_value, tokenizers, mode='train', separator_thickness=10):
     # Detect whether context is present
    if with_context:
        return collate_ctxt_and_tgt_fn(batch, padding_value, tokenizers, separator_thickness=separator_thickness)
    else:
        return collate_tgt_fn(batch, padding_value, tokenizers)

def collate_ctxt_and_tgt_fn(batch, padding_value, tokenizers, separator_thickness=10):
    all_texts = [item['all_texts'] for item in batch]
    all_images = [item['all_images'] for item in batch]
    all_images_masks = [item['all_images_masks'] for item in batch]
    all_font_paths = [item.get('font_path', []) for item in batch]
    writer_id = [item.get('writer_id', []) for item in batch]
    query_file = [item.get('query_file', []) for item in batch]
    query_img_original_widths = [item.get('query_img_original_widths', 0) for item in batch]

    padded_images_batch = []
    padding_masks_batch = []

    # --- 1️⃣ Determine the max width for right padding ---
    max_w = max(img.shape[-1] for images in all_images for img in images)

    # --- 2️⃣ Right-pad the context and target images ---
    for images, images_masks in zip(all_images, all_images_masks):
        context_images = images[:-1]
        context_masks = images_masks[:-1]
        target_image = images[-1]
        target_mask = images_masks[-1]

        padded_context = []
        padded_context_masks = []
        # Padding context
        for img, img_mask in zip(context_images, context_masks):
            C, h, w = img.shape
            pad_w = max_w - w
            padded = F.pad(img, (0, pad_w, 0, 0), value=padding_value)
            padded = torch.clamp(padded, 0, 1)
            padded_context.append(padded)

            # here pad = 1 and content = 0
            img_mask_padded = F.pad(img_mask, (0, pad_w, 0, 0), value=padding_value).to(torch.uint8)
            # flip the mask values of the context image:
            # 1 = context content, 0 = padding
            img_mask_padded = (img_mask_padded == 0).to(torch.uint8) * 1
            padded_context_masks.append(img_mask_padded)

        # Padding target
        C, h, w = target_image.shape
        pad_w = max_w - w
        padded_target = F.pad(target_image, (0, pad_w, 0, 0), value=padding_value)
        padded_target = torch.clamp(padded_target, 0, 1)

        # here pad = 1 (padding_value) and content = 0
        target_mask = F.pad(target_mask, (0, pad_w, 0, 0), value=padding_value)
        # flip the mask values of the query image:
        # 3 = query content, 0 = padding
        target_mask = (target_mask == 0).to(torch.uint8) * 3

        # Separator: value 2
        separator = torch.zeros((C, separator_thickness, max_w), dtype=padded_target.dtype)
        separator_mask = torch.full(
            (1, separator_thickness, max_w),
            fill_value=2,
            dtype=torch.uint8
        )

        # Concat vertical context + sep + target
        concat_img  = torch.cat(padded_context + [separator, padded_target], dim=1)
        concat_mask = torch.cat(padded_context_masks  + [separator_mask, target_mask], dim=1)

        padded_images_batch.append(concat_img)
        padding_masks_batch.append(concat_mask)

    # --- 3️⃣ Bottom padding so every item in the batch has the same height ---
    max_h = max(img.shape[1] for img in padded_images_batch)
    for i in range(len(padded_images_batch)):
        C, h, W = padded_images_batch[i].shape
        pad_h = max_h - h
        if pad_h > 0:
            padded_images_batch[i] = F.pad(padded_images_batch[i], (0, 0, 0, pad_h), value=padding_value)
            padding_masks_batch[i] = F.pad(padding_masks_batch[i], (0, 0, 0, pad_h), value=0)

    # --- 4️⃣ Stack final ---
    images_tensor = torch.stack(padded_images_batch, dim=0)
    images_masks  = torch.stack(padding_masks_batch, dim=0)
    context_texts = [text[0] for text in all_texts]
    target_texts  = [text[1] for text in all_texts]

    inputs = tokenizers.prepare_text_inputs(
        context_texts=context_texts,
        target_texts=target_texts,
        max_error_rate=0.2
    )
    inputs["images"]                    = images_tensor
    inputs["images_masks"]              = images_masks
    inputs["font_paths"]                = all_font_paths
    inputs["writer_id"]                 = writer_id
    inputs["query_file"]                = query_file
    inputs["context_strs"]              = context_texts
    inputs["query_img_original_widths"] = query_img_original_widths
    return inputs



def collate_tgt_fn(batch, padding_value, tokenizers):
    target_images = []
    target_texts = []

    masks = []

    # Find the max dimensions of the batch
    max_height = max(img.shape[-2] for item in batch for img in item["all_images"])
    max_width = max(img.shape[-1] for item in batch for img in item["all_images"])
    
    writer_id = [item.get('writer_id', []) for item in batch]
    query_file = [item.get('query_file', []) for item in batch]
    all_font_paths = [item.get('font_path', []) for item in batch]
    query_img_original_widths = [item.get('query_img_original_widths', 0) for item in batch]

    for item in batch:
        # Retrieve the last image (query)
        target_image = item["all_images"][-1]
        target_image_mask = item['all_images_masks'][-1]
        _, h, w = target_image.shape
        # Padding image
        pad_h = max_height - h
        pad_w = max_width - w
        padded_query = F.pad(target_image, (0, pad_w, 0, pad_h), value=padding_value)
        padded_query = torch.clamp(padded_query, 0, 1)
        
        # here pad = 1 (padding_value) and content = 0
        padded_query_mask = F.pad(target_image_mask, (0, pad_w, 0, pad_h), value=padding_value)
        # here pad = 0 (padding_value) and content = 3
        padded_query_mask = (padded_query_mask == 0).to(torch.uint8) * 3
        # Corresponding mask

        target_images.append(padded_query)
        masks.append(padded_query_mask)
        target_texts.append(item["all_texts"][-1])  # text associated with the query

    # Stacking
    images_tensor = torch.stack(target_images, dim=0)  # (B, C, H, W)
    images_masks = torch.stack(masks, dim=0)          # (B, 1, H, W)

    inputs = tokenizers.prepare_text_inputs(context_texts=None,
                                            target_texts=target_texts, 
                                            max_error_rate=0.2)
    inputs["images"] = images_tensor
    inputs["images_masks"] = images_masks
    inputs["font_paths"] = all_font_paths
    inputs["writer_id"] = writer_id
    inputs["query_file"] = query_file
    inputs["query_img_original_widths"] = query_img_original_widths
    return inputs


def collate_fn_factory(tokenizers, with_context):
    """
    Factory function to create a collate function with the given tokenizer and processor.
    """
    def tmp_collate_fn(batch):
        #return collate_fn(batch, 1.0, tokenizer, processor, config, mode) #ATTN padding_value
        return collate_fn(batch=batch, 
                          with_context=with_context,
                          padding_value=1.0, #ATTN padding_value
                          tokenizers=tokenizers,
                          ) 
    return tmp_collate_fn

@lru_cache(maxsize=1024)
def get_cached_font(font_path, font_size):
    return ImageFont.truetype(font_path, size=font_size)


class RealTextDataset(Dataset):
    """
    - Loads all .txt files at init
    - __getitem__ returns a randomly sampled text
    """

    def __init__(self, dataset_names):
        """
        base_path: root folder containing text files (recursively)
        """
        self.all_texts = {}
        for dataset_name in dataset_names:
            base_path = os.path.join(f"data/{dataset_name.lower()}_per_scripter_split_150DPI", "train")
            # ------------------------------------------------------------------
            # Load ALL text files into memory
            # ------------------------------------------------------------------
            self.all_texts[dataset_name] = []
            for root, _, files in os.walk(base_path):
                for file in files:
                    if file.lower().endswith(".txt"):
                        txt_path = os.path.join(root, file)
                        with open(txt_path, "r", encoding="utf-8") as f:
                            text = f.read().strip()
                            if text:
                                self.all_texts[dataset_name].append(text)

            if len(self.all_texts[dataset_name]) == 0:
                raise ValueError(f"No .txt files found in {base_path}")
            print(f"{len(self.all_texts[dataset_name])} text files loaded from {dataset_name}")

    def __len__(self):
        """
        Dataset length is defined as number of texts.
        Even though __getitem__ samples randomly, this keeps
        DataLoader behavior consistent.
        """
        return sum(len(texts) for texts in self.all_texts.values())

    def __getitem__(self, idx):
        """
        Randomly sample a text item.
        idx is ignored on purpose.
        """
        #random over the dictionary keys to choose a dataset, then random over that dataset's texts
        dataset_name = random.choice(list(self.all_texts.keys()))
        return random.choice(self.all_texts[dataset_name])

    def get_real_text(self, dataset_name=None):
        """
        Randomly sample a text item.
        idx is ignored on purpose.
        """
        return random.choice(self.all_texts[dataset_name])
