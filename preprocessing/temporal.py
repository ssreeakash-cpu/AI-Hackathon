import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

def pad_or_truncate_sequence(frames: List[np.ndarray], target_length: int, pad_value: float = 0.0) -> np.ndarray:
    """
    Takes a list of feature representations or images and forces them to a target sequence length.
    Critical for batching in LSTM / ConvLSTM temporal models.
    """
    if not frames:
        logger.warning("Empty sequence provided to pad_or_truncate_sequence.")
        return np.array([])

    seq_len = len(frames)
    frame_shape = frames[0].shape
    dtype = frames[0].dtype

    if seq_len == target_length:
        return np.stack(frames, axis=0)
    elif seq_len > target_length:
        # Truncate by taking the most recent frames (end of sequence)
        return np.stack(frames[-target_length:], axis=0)
    else:
        # Pre-pad with zeroes (or pad_value)
        padded = np.full((target_length, *frame_shape), fill_value=pad_value, dtype=dtype)
        padded[-seq_len:] = np.stack(frames, axis=0)
        return padded

def create_temporal_mask(sequence_length: int, valid_length: int) -> np.ndarray:
    """
    Creates a boolean mask indicating valid timesteps vs padded timesteps.
    (True for valid, False for padding). Useful for attention mechanisms.
    """
    mask = np.zeros((sequence_length,), dtype=bool)
    if valid_length > 0:
        # Assumes pre-padding
        start_idx = max(0, sequence_length - valid_length)
        mask[start_idx:] = True
    return mask
