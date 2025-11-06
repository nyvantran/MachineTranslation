import torch
from old_core.model import Transformer
import os
import logging
import datetime
import errno
import json
import shutil


def save_model(model, epoch, loss, config, save_dir='checkpoints'):
    """Saves the model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        loss (float): The current loss value.
        config (dict): Configuration dictionary.
        save_dir (str): Directory to save the checkpoint.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'config': config
    }

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'checkpoint_epoch_{epoch}_{timestamp}.pth'
    filepath = os.path.join(save_dir, filename)

    torch.save(checkpoint, filepath)
    logging.info(f'Model saved to {filepath}')


def load_model(model, filepath):
    """Loads the model and optimizer state from a checkpoint file.

    Args:a
        model (torch.nn.Module): The model to lod the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        filepath (str): Path to the checkpoint file.
    Returns:
        epoch (int): The epoch number from the checkpoint.
        loss (float): The loss value from the checkpoint.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)

    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    logging.info(f'Model loaded from {filepath}, epoch: {epoch}, loss: {loss}')
    return epoch, loss
