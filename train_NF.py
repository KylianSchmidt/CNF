import argparse
from typing import Literal

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

from lib.NormFlow import NormalizingFlowModel
from lib.data_utils import createJetData
from tabulate import tabulate


class Args:
    root_dir: str
    checkpoint_path: str
    c: float
    channel: Literal["signal", "background"]
    num_jets: Literal[0, 1, 2]


def prepare_dataloader(args: Args):
    # Create jet data using the provided utility
    jet_data, jet_detailed_label, _, _ = createJetData(
        args.num_jets,
        False,
        seed=78,
        root_dir=args.root_dir,
    )

    # Convert data to tensors
    S_tensor = torch.tensor(jet_data[jet_detailed_label == 1], dtype=torch.float32)
    BG_tensor = torch.tensor(jet_data[jet_detailed_label == 0], dtype=torch.float32)

    # Equalize dataset size between signal and background
    max_size = np.min([len(S_tensor), len(BG_tensor)])
    dataset: TensorDataset = {
        "signal": TensorDataset(S_tensor[:max_size], BG_tensor[:max_size]),
        "background": TensorDataset(BG_tensor[:max_size], S_tensor[:max_size]),
    }[args.channel]

    # Split dataset into training and validation sets (80/20 split)
    n_val = int(0.2 * len(dataset))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    batch_size = 1000
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Calculate mean and standard deviation of the signal tensor
    X_mean = torch.mean(S_tensor, dim=0)
    X_std = torch.std(S_tensor, dim=0)

    return train_loader, val_loader, X_mean, X_std


def main(args: Args):
    # Setup root directory and add necessary paths
    root_dir = args.root_dir
    print("HEP Challenge Root directory is", root_dir)

    train_loader, val_loader, X_mean, X_std = prepare_dataloader(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    # Setup callbacks: early stopping and model checkpointing
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=100, mode="min"
    )
    file_name: str = {
        "signal": f"NF_njets_{args.num_jets}_signal_c_{args.c}",
        "background": f"NF_njets_{args.num_jets}_background_c_{args.c}",
    }[args.channel]

    input_dim = {
        1: 20,  # Case: 1 jet
        2: 27,  # Case: 2 jets
    }[args.num_jets]

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        filename=file_name
    )
    bg_model = NormalizingFlowModel(
        c=args.c,
        input_dim=input_dim,
        clamp_val=-50,
        n_layers=30,
        X_mean=X_mean,
        X_std=X_std,
    )

    # Create the trainer and fit the model
    trainer = pl.Trainer(
        max_epochs=500,
        default_root_dir=args.checkpoint_path,
        log_every_n_steps=10,
        accelerator=device,
        devices=[1],
        callbacks=[checkpoint_callback, early_stop_callback]
    )
    trainer.fit(bg_model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Normalizing Flow Model")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="../HEP-Challenge/",
        help="Root directory containing input_data, ingestion_program, and scoring_program folders"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="Test/",
        help="Path to a model checkpoint to load (optional)"
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--channel",
        type=str,
        choices=["signal", "background"],
        default="signal",
    )
    parser.add_argument(
        "--num_jets",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    args_dict = vars(args)
    print(tabulate(args_dict.items(), headers=["Argument", "Value"], tablefmt="simple"))
    main(args)
