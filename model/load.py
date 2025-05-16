# Copyright 2023 BioMap (Beijing) Intelligence Technology Limited

import math
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from pretrainmodels import select_model


def next_16x(x):
    return int(math.ceil(x / 16) * 16)


def seed_all(seed, cuda_deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def gatherDatanopad(data, labels, pad_token_id):
    max_num = labels.sum(1)
    none_labels = ~labels
    labels = labels.float()
    labels[none_labels] = torch.tensor(-float("Inf"), device=labels.device)

    tmp_data = torch.tensor(
        [(i + 1) * 20000 for i in range(labels.shape[1], 0, -1)], device=labels.device
    )
    labels += tmp_data

    fake_label_gene_idx = labels.topk(max_num).indices

    new_data = torch.gather(data, 1, fake_label_gene_idx)

    padding_labels = new_data == pad_token_id

    return new_data, padding_labels


def gatherData(data, labels, pad_token_id):
    value_nums = labels.sum(
        1
    )  # Number of non-zero elements for each sample in the batch

    # --- START MODIFICATION ---
    ENCODER_TOKEN_LIMIT = 13000  # START WITH THIS VALUE.
    # Try 10000 if this is too restrictive,
    # or 5000 if 7000 still OOMs.
    # This limits the sequence length for the encoder.
    max_num_from_data = max(
        value_nums
    )  # Max non-zero in current batch (batch_size=1 for you)

    # Ensure max_num is an integer for later tensor operations
    # and handle potential tensor nature of max_num_from_data
    if torch.is_tensor(max_num_from_data):
        current_max_val = max_num_from_data.item()
    else:  # Assuming it's already a Python number (e.g., int)
        current_max_val = max_num_from_data

    max_num = min(current_max_val, ENCODER_TOKEN_LIMIT)

    if current_max_val > ENCODER_TOKEN_LIMIT:
        # This print might be very verbose if many cells are capped.
        # Consider removing or reducing frequency if it clutters logs too much.
        print(
            f"INFO: gatherData - Capping effective sequence length from {current_max_val} to {max_num} for this sample."
        )
    # --- END MODIFICATION ---

    # Original padding logic to new max_num
    fake_data = torch.full((data.shape[0], max_num), pad_token_id, device=data.device)
    data_padded_to_max_num = torch.hstack(
        [data, fake_data]
    )  # Pad original data up to max_num if needed (conceptually)

    # Original logic for selecting tokens based on labels, but now with a capped max_num
    # The labels indicate which are actual values vs padding for selection.
    # The `topk(max_num)` will select up to `max_num` actual non-zero tokens if available,
    # or fewer if the sample has less than `max_num` non-zero tokens.
    # If a sample has > max_num non-zero tokens, topk will effectively truncate.

    fake_label_for_padding = torch.full(
        (labels.shape[0], max_num),
        1,  # Should be 1 for actual data, 0 for padding later
        device=labels.device,
    )
    none_labels = ~labels  # Where original labels were False (zeroes)
    labels_float = labels.float()
    labels_float[none_labels] = torch.tensor(
        -float("Inf"), device=labels.device
    )  # Make zeroes highly negative for topk

    # Add a large value scaled by inverse position to prioritize earlier genes in case of ties / truncation
    # This ensures that if truncation happens due to ENCODER_TOKEN_LIMIT, it's not entirely random
    # among genes with the same expression value (though value_labels is just >0).
    # For >0 selection, this tie-breaking is less critical than for value-based topk.
    tmp_data_for_sorting = torch.tensor(
        [(i + 1) * 20000 for i in range(labels_float.shape[1], 0, -1)],
        device=labels_float.device,
    )
    labels_for_sort = labels_float + tmp_data_for_sorting

    # Conceptually pad labels_for_sort if its width is less than max_num (though labels.shape[1] is 19k+)
    # The key is that topk will select from the original 19k+ genes based on value_labels
    # and then the resulting sequence will be of length max_num.

    # This line effectively selects the indices of the top 'max_num' elements based on 'labels_for_sort'
    # where 'labels' (from value_labels) filters for non-zero.
    selected_indices = labels_for_sort.topk(
        max_num
    ).indices  # This will select up to max_num tokens

    # Gather the actual data values using these selected indices
    new_data = torch.gather(data, 1, selected_indices)

    # Determine padding based on whether the gathered data is the pad_token_id
    # This happens if a sample had fewer than max_num non-zero tokens
    # and was padded by gatherData internally if selected_indices pointed to padded parts (unlikely with topk from original data range)
    # More accurately, padding_labels should reflect if the selected token was originally a pad_token_id *or if the gene was zero initially*.
    # However, the model's encoder expects padding_mask for the *final* sequence of length `max_num`.
    # The original `gatherData` logic seemed to create `padding_labels = (new_data == pad_token_id)`.
    # Let's refine the padding_labels to reflect actual non-zero elements up to max_num.
    # Number of actual non-zero elements for each sample (up to max_num)
    actual_non_zeros_in_selected = torch.gather(
        value_nums, 0, torch.arange(data.shape[0], device=data.device)
    ).clamp_max(max_num)

    padding_labels = torch.ones_like(
        new_data, dtype=torch.bool
    )  # Start with all True (padded)
    for i in range(padding_labels.shape[0]):  # For each sample in batch
        padding_labels[i, : actual_non_zeros_in_selected[i]] = (
            False  # Mark actual data as not padded
        )

    return new_data, padding_labels


def convertconfig(ckpt):
    newconfig = {}
    newconfig["config"] = {}
    model_type = ckpt["config"]["model"]

    for key, val in ckpt["config"]["model_config"][model_type].items():
        newconfig["config"][key] = val

    for key, val in ckpt["config"]["dataset_config"]["rnaseq"].items():
        newconfig["config"][key] = val

    if model_type == "performergau_resolution":
        model_type = "performer_gau"

    import collections

    d = collections.OrderedDict()
    for key, val in ckpt["state_dict"].items():
        d[str(key).split("model.")[1]] = val

    newconfig["config"]["model_type"] = model_type
    newconfig["model_state_dict"] = d
    newconfig["config"]["pos_embed"] = False
    newconfig["config"]["device"] = "cuda"
    return newconfig


def load_model(best_ckpt_path, device):
    model_data = torch.load(best_ckpt_path, map_location=device)
    if not model_data.__contains__("config"):
        print("***** No config *****")
        config = {}
        config["model_type"] = "flash_all"
    else:
        config = model_data["config"]
        print(config)
    if not config.__contains__("qv_dim"):
        if config["model"] != "mae_autobin":
            if config.__contains__("dim_head"):
                config["qv_dim"] = config["dim_head"]
            else:
                print("***** No qv_dim ***** set 64")
                config["qv_dim"] = 64
    if not config.__contains__("ppi_edge"):
        config["ppi_edge"] = None
    model = select_model(config)
    model_state_dict = model_data["model_state_dict"]
    model.load_state_dict(model_state_dict)
    return model.cuda(), config


def load_model_frommmf(best_ckpt_path, key="gene"):
    # Determine the device to load onto
    if torch.cuda.is_available():
        # Let PyTorch automatically handle the device based on SLURM allocation
        # (assuming CUDA_VISIBLE_DEVICES is set correctly by SLURM)
        map_location = torch.device("cuda")
        # Or, more explicitly if needed, though usually not necessary with SLURM:
        # map_location = f"cuda:{torch.cuda.current_device()}"
    else:
        # Fallback to CPU if needed, though the error message indicates
        # you *expect* CUDA but it's not found by PyTorch in the job.
        map_location = torch.device("cpu")
        print("Warning: CUDA not available, loading model to CPU.")

    model_data = torch.load(best_ckpt_path, map_location=map_location)
    model_data = model_data[key]
    model_data = convertconfig(model_data)
    if not model_data.__contains__("config"):
        print("***** No config *****")
        config = {}
        config["model_type"] = "flash_all"
    else:
        config = model_data["config"]
        print(config)
    if not config.__contains__("qv_dim"):
        if config["model"] != "mae_autobin":
            if config.__contains__("dim_head"):
                config["qv_dim"] = config["dim_head"]
            else:
                print("***** No qv_dim ***** set 64")
                config["qv_dim"] = 64
    if not config.__contains__("ppi_edge"):
        config["ppi_edge"] = None
    model = select_model(config)
    model_state_dict = model_data["model_state_dict"]
    model.load_state_dict(model_state_dict)
    return model.cuda(), config


def main_gene_selection(X_df, gene_list):
    """
    Describe:
        rebuild the input adata to select target genes encode protein
    Parameters:
        adata->`~anndata.AnnData` object: adata with var index_name by gene symbol
        gene_list->list: wanted target gene
    Returns:
        adata_new->`~anndata.AnnData` object
        to_fill_columns->list: zero padding gene
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    padding_df = pd.DataFrame(
        np.zeros((X_df.shape[0], len(to_fill_columns))),
        columns=to_fill_columns,
        index=X_df.index,
    )
    X_df = pd.DataFrame(
        np.concatenate([df.values for df in [X_df, padding_df]], axis=1),
        index=X_df.index,
        columns=list(X_df.columns) + list(padding_df.columns),
    )
    X_df = X_df[gene_list]

    var = pd.DataFrame(index=X_df.columns)
    var["mask"] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns


def get_genename():
    gene_list_df = pd.read_csv(
        "./OS_scRNA_gene_index.19264.tsv", header=0, delimiter="\t"
    )
    return list(gene_list_df["gene_name"])


def save_AUCs(AUCs, filename):
    with open(filename, "a") as f:
        f.write("\t".join(map(str, AUCs)) + "\n")


def getEncoerDecoderData(data, data_raw, config):
    decoder_data = data.clone().detach()
    decoder_data_padding = torch.full_like(data, False, dtype=torch.bool).to(
        data.device
    )

    encoder_data_labels = data_raw > 0
    encoder_data, encoder_data_padding = gatherData(
        decoder_data, encoder_data_labels, config["pad_token_id"]
    )
    new_data_raw = data_raw
    data_gene_ids = torch.arange(data.shape[1], device=data.device).repeat(
        data.shape[0], 1
    )
    encoder_position_gene_ids, _ = gatherData(
        data_gene_ids, encoder_data_labels, config["pad_token_id"]
    )
    decoder_position_gene_ids = data_gene_ids
    data_mask_labels = None

    encoder_position_gene_ids[encoder_data_padding] = config["seq_len"]
    decoder_position_gene_ids[decoder_data_padding] = config["seq_len"]

    return (
        encoder_data,
        encoder_position_gene_ids,
        encoder_data_padding,
        encoder_data_labels,
        decoder_data,
        decoder_data_padding,
        new_data_raw,
        data_mask_labels,
        decoder_position_gene_ids,
    )
