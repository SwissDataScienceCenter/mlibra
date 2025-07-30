"""
config.py
Configuration for the MALDI experiment.
"""
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import json

@dataclass
class MaldiConfig:
    mode: str
    dataset_path: Path
    maldi_file: Path
    exp_name: str
    available_lipids_file: str
    output_path: Path
    exp_path: Path
    checkpoint_path: Path
    num_inducing: int
    seed: int
    epochs: int
    latent_dim: int
    device: str
    kernel: str
    slices_dataset_file: str
    log_transform: bool
    nu: float
    n_pixels: int
    batch_size: int
    learning_rate: float
    section_filter: List[Tuple[int, int]]
    test_filter: List[Tuple[int, int]]
    selected_lipids_names: List[str]
    available_lipids: List[str]
    selected_channels: List[int]

    @staticmethod
    def from_args(args):
        """
        Create an instance of MaldiConfig from command line arguments.

        By creating the necessary directories and loading the arguments.

        Args:
            args (dict): Dictionary containing the experiment arguments, read from command line or config file.
        """
        logging.info("Creating experiment configuration from arguments")
        mode = args["mode"]
        # Extracting path arguments
        dataset_path = Path(args["dataset_path"])
        maldi_file = Path(args["maldi_file"])
        exp_name = args["exp_name"]
        available_lipids_file = args["available_lipids_file"]
        selected_lipids_file = args.get("selected_lipids_file",available_lipids_file)
        # Read selected lipids names
        available_lipids, selected_channels, selected_lipids_names = read_channels(selected_lipids_file, available_lipids_file)
        output_path = Path(args["output_dir"])
        slices_dataset_file = args['slices_dataset_file']
        # Train test split arguments
        section_filter, test_filter = extract_filters(slices_dataset_file)
        # GP arguments
        num_inducing = args["num_inducing"]
        kernel = args['kernel']
        seed = random.randint(0, 20000) if args["seed"] == -1 else args["seed"]
        latent_dim = args['latent_dim']
        log_transform = args['log_transform']
        nu = args.get("nu", 1.5)  # Default value for nu if not provided
        n_pixels = args.get("n_pixels", 10)
        # Optimization arguments
        epochs = args['epochs']
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = args['batch_size']
        learning_rate = args.get("learning_rate", 0.001)
        exp_name = exp_name + str(args["n_pixels"])

        # Create paths
        if log_transform:
            exp_name = exp_name + "_log"
        # with the experiment name we create the folders for the experiment
        exp_path = output_path / exp_name
        logging.info("Creating experiment path: " + str(exp_path))
        exp_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Experiment path: {exp_path} created")
        # Create the path for the checkpoints
        checkpoint_path = exp_path / "checkpoints"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        # Load arguments if previously saved
        if exp_path.exists() and (exp_path / "args.npy").exists():
            logging.info(f"Experiment {exp_path} already exists")
            logging.info("ATTENTION: loading original experiment arguments and using them")
            args = np.load(exp_path / "args.npy", allow_pickle=True).item()
        else:
            logging.info(f"Creating experiment {exp_path}")
            exp_path.mkdir(parents=True, exist_ok=True)
            np.save(exp_path / "args.npy", args)

        return MaldiConfig(mode=mode,
                           dataset_path=dataset_path,
                           maldi_file=maldi_file,
                           exp_name=exp_name,
                           available_lipids_file=available_lipids_file,
                           selected_lipids_names=selected_lipids_names,
                           output_path=output_path,
                           exp_path=exp_path,
                           checkpoint_path=checkpoint_path,
                           num_inducing=num_inducing,
                           batch_size=batch_size,
                           seed=seed,
                           epochs=epochs,
                           latent_dim=latent_dim,
                           device=device,
                           kernel=kernel,
                           slices_dataset_file=slices_dataset_file,
                           log_transform=log_transform,
                           nu=nu,
                           n_pixels=n_pixels,
                           learning_rate=learning_rate,
                           section_filter=section_filter,
                           test_filter=test_filter,
                           selected_channels=selected_channels,
                           available_lipids=available_lipids)

    def to_dict(self):
        """
        Convert the configuration to a dictionary.

        Returns:
            dict: Dictionary representation of the configuration.
        """
        return {
            "mode": self.mode,
            "dataset_path": str(self.dataset_path),
            "maldi_file": str(self.maldi_file),
            "exp_name": self.exp_name,
            "available_lipids_file": self.available_lipids_file,
            "output_path": str(self.output_path),
            "exp_path": str(self.exp_path),
            "checkpoint_path": str(self.checkpoint_path),
            "num_inducing": self.num_inducing,
            "seed": self.seed,
            "epochs": self.epochs,
            "latent_dim": self.latent_dim,
            "device": self.device,
            "kernel": self.kernel,
            "slices_dataset_file": self.slices_dataset_file,
            "log_transform": self.log_transform,
            "nu": self.nu,
            "n_pixels": self.n_pixels,
            "learning_rate": self.learning_rate,
            "section_filter": self.section_filter,
            "test_filter": self.test_filter,
            "selected_lipids_names": self.selected_lipids_names
        }

def extract_filters(left_out_slice):
    # left out slice is a list of dictionary entries with sample and section, that is read from a json file.
    # we will load everything and then discard the sample-section query.
    left_out_slice = left_out_slice
    # make sure the left_out_slices file is json
    assert left_out_slice.endswith(".json"), "The left out slices file should be a json file"
    left_out_slices = left_out_slice
    logging.info(f"loading left out slices from {left_out_slices}")
    with open(left_out_slices, 'r') as file:
        filters = json.load(file)
    # we load the filters
    train_filter = filters["train"]
    test_filter = filters["test"]
    ignore_filter = filters["ignore"]
    # now each filter contains a list of lists, I want to convert them to a list of tuples
    train_filter = [[tuple(i)] for i in train_filter]
    test_filter = [[tuple(i)] for i in test_filter]
    ignore_filter = [[tuple(i)] for i in ignore_filter]
    # we load the maldi file
    logging.info("slices train: " + str(train_filter))
    logging.info("slices test: " + str(test_filter))
    logging.info("slices ignore: " + str(ignore_filter))
    return train_filter, test_filter

def read_channels(selected_channels, available_channels):
    available_channels = np.load(available_channels, allow_pickle=True)
    ascii_line_string = "---------------------------------"
    print(ascii_line_string)
    print("Available channels:")
    print(available_channels)
    print(ascii_line_string)
    print(ascii_line_string)
    print("Selected channels:")
    # we select sm channels
    # we find the position of the selected channels names in available channels
    selected_channels_names = np.load(selected_channels, allow_pickle=True)
    selected_channels = [i for i, v in enumerate(available_channels) if v in selected_channels_names]

    print(available_channels[selected_channels])
    print(ascii_line_string)
    assert len(selected_channels) > 0, "No channels selected"
    logging.info(f"loading channels: {available_channels[selected_channels]}")
    return available_channels, selected_channels, selected_channels_names
