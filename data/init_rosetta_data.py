import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
from data.synth_img_generator import SynthImageTextDataset, collate_fn_factory
from data.real_icl_dataset import FewShotWriterDataset


def init_dataloaders(config, tokenizers):
    # --- Dataloaders initialization ---

    val_synth_loaders = {}
    valid_real_loader = None

    real_dataset = config.data_config.real_dataset
    dataset_path = f'data/{real_dataset.lower()}_per_scripter_split_150DPI'

    is_only_data_synth = config.data_config.end_p_synth ==1.0

    if config.data_config.start_p_synth > 0:
        if "synth_fonts" in config.data_config.synth_datasets:
            val_synth_loaders['synth_fonts'] = init_online_dataloader(config.data_config,
                                                    tokenizers=tokenizers,
                                                    mode='generation')

    if config.data_config.start_p_synth != 1.0:
        valid_real_loader = init_real_dataloader(config.data_config,
                                            tokenizers=tokenizers,
                                            dataset_path=dataset_path,
                                            mode='generation',
                                            set='valid')

    if is_only_data_synth:
        if "synth_fonts" in config.data_config.synth_datasets:
            train_loader = init_online_dataloader(config.data_config,
                                                tokenizers=tokenizers,
                                                mode='train')
        print("Training only on synthetic data")

    else:
        train_synth_loaders = []
        if "synth_fonts" in config.data_config.synth_datasets:
            train_synth_loaders.append(init_online_dataloader(config.data_config,
                                                tokenizers=tokenizers,
                                                mode='train').dataset)

        train_synth_dataset = init_mixed_dataloader_synth(config.data_config,
                                                        train_synth_loaders,
                                                        tokenizers=tokenizers).dataset

        train_loader = init_mixed_dataloader_simple(config.data_config,
                                                synth_dataset=train_synth_dataset,
                                                tokenizers=tokenizers,
                                                dataset_path=dataset_path,
                                                mode='train',
                                                set='train',
                                                )

    return train_loader, valid_real_loader, val_synth_loaders


def init_online_dataloader(config, tokenizers, mode='train'):
    if mode == 'train':
        font_folder = config.train_font_path
        use_image_augmentation = config.use_image_augmentation
    elif mode == 'generation':
        font_folder = config.valid_font_path
        use_image_augmentation = False
    font_names = sorted(os.listdir(font_folder))
    font_paths = [os.path.join(font_folder,  name) for name in font_names]
    

    dataset = SynthImageTextDataset(
        num_context_lines=config.start_num_context_lines,
        padding_value=1,
        font_folder=font_folder,
        font_paths=font_paths,
        context_content=config.context_content,
        target_content=config.target_content,
        imgs_per_epoch=9984 if mode=='train' else 1000, 
        image_augmentation=use_image_augmentation,
        charset=config.charset,
        )
    
    with_context = config.max_num_context_lines > 0
    dataloader = DataLoader(dataset,
                            batch_size=config.train_batch_size if mode=='train' else config.valid_batch_size,
                            shuffle=True if mode=='train' else False,
                            collate_fn=collate_fn_factory(tokenizers=tokenizers,
                                                          with_context=with_context),
                            num_workers=8,
                            pin_memory=True) #

    return dataloader


def init_real_dataloader(config, tokenizers, dataset_path, mode='train', set='train'):
    use_image_augm = config.use_image_augmentation
    if mode != 'train':
        use_image_augm = False

    dataset = FewShotWriterDataset(base_path = f"{dataset_path}/{set}",
                                    num_context_lines=config.start_num_context_lines,
                                    use_image_aug=use_image_augm,
                                    mode=mode
                                    )
    
    batch_size = config.train_batch_size if mode=='train' else config.valid_batch_size


    with_context = config.max_num_context_lines > 0
    dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True if mode=='train' else False,
                        collate_fn=collate_fn_factory(tokenizers=tokenizers,
                                                        with_context=with_context),   
                        num_workers=8 if mode=='train' else 0,
                        pin_memory=True if mode=='train' else False)
    return dataloader

class MixedDataset(Dataset):
    def __init__(self, real_dataset, synth_dataset, p_synth=0.25, length=10000):
        self.real_dataset = real_dataset
        self.synth_dataset = synth_dataset
        self.length = length
        self.p_synth = p_synth  # stores the modifiable probability

    def set_p_synth(self, new_p):
        """Updates the probability of choosing a synthetic sample."""
        self.p_synth = max(0.0, min(1.0, new_p))  # clamp between 0 and 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Decide on the fly
        if np.random.rand() < self.p_synth:
            synth_idx = np.random.randint(len(self.synth_dataset))
            return self.synth_dataset[synth_idx]
        else:
            real_idx = np.random.randint(len(self.real_dataset))
            return self.real_dataset[real_idx]


class MultiMixedDataset(Dataset):
    def __init__(self, synth_datasets, length=10000):
        self.length = length
        self.synth_datasets = synth_datasets

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Decide on the fly
        i = np.random.randint(len(self.synth_datasets))
        synth_idx = np.random.randint(len(self.synth_datasets[i]))
        return self.synth_datasets[i][synth_idx]
 

def init_mixed_dataloader_simple(config, tokenizers, synth_dataset, dataset_path, mode='train', set='train'):
    """
    mode : train or generation
    set : train, valid, test
    """
    p_synth = config.start_p_synth
 
    real_dataset = init_real_dataloader(config, 
                                        tokenizers=tokenizers,
                                        dataset_path=dataset_path,
                                        mode=mode, 
                                        set=set).dataset

    # Combine into MixedDataset
    mixed_dataset = MixedDataset(real_dataset=real_dataset, synth_dataset=synth_dataset, p_synth=p_synth)

    with_context = config.max_num_context_lines > 0

    # Standard dataloader
    dataloader = DataLoader(
        mixed_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn_factory(tokenizers=tokenizers,
                                      with_context=with_context),
        num_workers=8, # 8
        pin_memory=True
    )

    return dataloader


def init_mixed_dataloader_synth(config, synth_datasets, tokenizers):
    """
    mode : train or generation
    set : train, valid, test
    """

    if len(synth_datasets) > 1:
        dataset = MultiMixedDataset(synth_datasets=synth_datasets)
    else: 
        dataset = synth_datasets[0]
    
    with_context = config.max_num_context_lines > 0
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn_factory(tokenizers=tokenizers,
                                      with_context=with_context),
        num_workers=8,
        pin_memory=True
    )
    return dataloader