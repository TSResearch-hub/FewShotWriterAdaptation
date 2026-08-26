data_config = {
    "real_dataset": "IAM", #RIMES or IAM or CVL
    "synth_datasets":["synth_fonts"], 
    
    "start_num_context_lines": 1, # Max number of image/text pairs to include in the context
    "max_num_context_lines": 12,  # Max number of image/text pairs to include in the context
    
    "train_batch_size": 1, # Batch size during training
    "valid_batch_size": 1, # Batch size during validation

    "start_p_synth" : 0.9,
    "end_p_synth" : 0.2,
    "start_epoch_p_synth" : 0, 
    "end_epoch_p_synth" : 500, 
    "use_image_augmentation": True,   # Whether to apply image augmentation (e.g., Gaussian noise, color jitter)
    
    "charset": " !\"#%&'()*+,-./0123456789:;<=>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_abcdefghijklmnopqrstuvwxyz{}£€",
    "target_content": ["IAM"],
    "context_content": ["IAM"],
    "font_path": "data/fonts/",  # Path to all available fonts
    "train_font_path": "data/fonts/train_script_fonts", #train_fonts #train_fonts  # Path to fonts used specifically for training
    "valid_font_path": "data/fonts/valid_script_fonts", # Path to fonts used specifically for validation

}

train_config = {
    # === Training parameters ===
    "num_epochs": 10000,            # Total number of training epochs
    "learning_rate": 1e-4, #1e-4 # Learning rate for the optimizer (default was 5e-5)
    # === Validation settings ===
    "valid_interval": 10, #10        # Perform validation every N epochs
    "valid_cer_threashold": 0.05, #if the real valid is lower than the threashold : num_context_lines +=1
    "best_metric" : "cers",  # or "ters"
    "load_checkpoint_path": 'checkpoints/ICDAR_icl_12MaxCtxtLines_9IAM_90synth_fonts__FTscript_realTxt_fcn_256dim_8ICLlayers_8ICLheads_relativeCAT_ImgAug_PosStatic_loaded.pt', 
    "use_wandb": False, # Whether to use Weights & Biases for experiment tracking 
}

model_config = {
    "training_tasks" : ["icl"],  #["icl", "ocr"]
    "pos_enc" : "static", # "static" or "learnable"
    "is_cnn_ctc" : False, # Whether to use a CNN+CTC architecture for the OCR decoder (instead of a transformer decoder)
    "encoder" : {
        "dim_model": 256,
         # === Dropout curriculum ===
        "start_dropout_rate" :0.0,
        "end_dropout_rate" : 0.5,
        "start_epoch_dropout" : 0, 
        "end_epoch_dropout" : 1000, 
    },

    "decoders" : {
        "icl" : {
            "is_autoreg" : True,  # NAR: Non-Autoregressive
            "num_layers" : 8, 
            "num_heads" : 8, 
            "dim_model" : 256,
            "dim_ffwd" : 512,
            "vocab_size" : 70,
            "with_context" : True,
            "lr" :  1e-4,
            "mapping" : "relative", # "relative" or "random"

        },
        "ocr" : {
            "is_autoreg" : True,  # NAR: Non-Autoregressive
            "num_layers" : 8,
            "num_heads" : 8,
            "dim_model" : 256,
            "dim_ffwd" : 512,
            "with_context" : False,
            "lr" :  1e-4,
        },

    },
 
    "max_seq_len" : 3000, #used for the positional encoding of context + query
}

params = {
    "train_config": train_config,
    "data_config": data_config,
    "model_config": model_config,
}
