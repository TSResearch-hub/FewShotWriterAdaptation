"""
Shared utility functions: configuration, checkpoint naming, and CTC decoding.
"""
import torch


class RosettaConfigDict(dict):
    """Configuration dictionary allowing attribute-style access to keys (config.model_config.xxx)."""

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            return RosettaConfigDict(value)
        return value


def define_checkpoint_name(config):
    """Builds a readable checkpoint/experiment name from the configuration."""
    data_cfg = config.data_config
    train_cfg = config.train_config
    model_cfg = config.model_config

    parts = []
    parts.extend(model_cfg.training_tasks)
    if model_cfg.is_cnn_ctc:
        parts.append("CNNCTC")

    if "ocr" in model_cfg.training_tasks:
        if model_cfg.decoders.ocr.is_autoreg:
            parts.append("TrAutoreg")
        else:
            parts.append("TrCTC")

    if "icl" in model_cfg.training_tasks:
        if model_cfg.decoders.icl.is_autoreg:
            parts.append("TrAutoreg")
        else:
            parts.append("TrCTC")

        parts.append(f"{model_cfg.decoders.icl.num_layers}ICLlayers")
        parts.append(f"{model_cfg.decoders.icl.num_heads}ICLheads")
        parts.append(f"{model_cfg.decoders.icl.mapping}CAT")

    parts.append(f"{config.data_config.max_num_context_lines}MaxCtxtLines")

    if data_cfg.start_p_synth < 1.0:
        real_data_percent = int((1 - data_cfg.start_p_synth) * 100)
        parts.append(f"{real_data_percent}{data_cfg.real_dataset}")

    if data_cfg.start_p_synth > 0:
        synth_data_ratio = int(data_cfg.start_p_synth * 100)
        synth_datasets_str = "_".join(data_cfg.synth_datasets)
        parts.append(f"{synth_data_ratio}{synth_datasets_str}")

        if "synth_fonts" in data_cfg.synth_datasets:
            parts.append("FTall" if data_cfg.train_font_path == "data/fonts/train_fonts" else "FTscript")


        elif len(data_cfg.target_content) > 1 and len(data_cfg.context_content) > 1:
            parts.append("multiDataSynth")

    if data_cfg.use_image_augmentation:
        parts.append("ImgAug")
    else:
        parts.append("NOImgAug")

    if model_cfg.pos_enc == "static":
        parts.append("PosStatic")
    else:
        parts.append("PosLearn")

    parts.append("loaded" if train_cfg.load_checkpoint_path is not None else "scratch")

    return "_".join(parts)

def ctc_greedy_decode(logits, blank_id):
    pred = logits.argmax(dim=-1)
    return [
        s[s != blank_id].tolist()
        for s in map(torch.unique_consecutive, pred)
    ]
