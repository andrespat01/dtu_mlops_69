import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from data import tweets
import typer
import wandb
from transformers import AutoModel
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import OmegaConf
from pathlib import Path

"""
Found this github project:
Fine-Tuning BERT for Tweet Classification

https://gist.github.com/alexandster/211909d55fffae4efbd216e5a01b338e
"""


class Model(pl.LightningModule):
    """Lightning module"""

    def __init__(self, bert: AutoModel, lr: int) -> None:
        """Initialize the model.
        
        args: bert, lr
        """
        super(Model, self).__init__()

        self.bert = bert
        self.lr = lr

        # Freeze all layers of BERT to speed up training
        for name, param in self.named_parameters():
            if "encoder.layer" in name:
                param.requires_grad = False  # Freezing the layers

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768, 512)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512, 2)

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        args: sent_id, mask
        returns: torch.Tensor
        """

        # pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask)

        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)

        return x

    def training_step(self, batch, batch_idx):
        """Training step.
        
        args: batch, batch_idx
        """
        sent_id, mask, targets = (
            batch  # Unpack the batch into input_ids, attention_mask, and target
        )
        outputs = self(sent_id, mask)  # Pass the inputs to the model
        loss = nn.CrossEntropyLoss()(outputs, targets)  # Compute the loss

        # Calculate accuracy
        acc = (outputs.argmax(dim=1) == targets).float().mean()
        self.log("train_loss", loss)  # Log the training loss
        self.log("train_acc", acc)  # Log the training accuracy

        if batch_idx % 100 == 0:
            print(
                f"Step {batch_idx}: train_loss = {loss.item()}, train_acc = {acc.item()}"
            )
        return loss

    def validation_step(self, batch) -> None:
        """Validation step.
        
        args: batch
        """
        sent_id, mask, targets = (
            batch  # Unpack the batch into input_ids, attention_mask, and target
        )
        outputs = self(sent_id, mask)  # Pass the inputs to the model
        loss = nn.CrossEntropyLoss()(outputs, targets)  # Compute the loss
        # Calculate accuracy
        acc = (outputs.argmax(dim=1) == targets).float().mean()
        self.log("val_loss", loss, on_epoch=True)  # Log the validation loss
        self.log("val_acc", acc, on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer.
        
        returns: torch.optim.Optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self, batch_size: int) -> torch.utils.data.DataLoader:
        """ Training dataloader
        
        args: batch_size: int
        returns: torch.utils.data.DataLoader
        """
        train_set, _ = tweets()  # Ensure this is returning a valid dataset
        return torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, num_workers=4, pin_memory=True
        )

    def val_dataloader(self, batch_size: int) -> torch.utils.data.DataLoader:
        """ Validation dataloader 
        
        args: batch_size: int
        returns: torch.utils.data.DataLoader
        """
        _, val_set = tweets()  # Assuming corrupt_mnist returns a validation dataset
        return torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, num_workers=4, pin_memory=True
        )


def train(lr: float = None, batch_size: int = None, epochs: int = None, config_path: str = None) -> None:
    """Train the model."""
    # Using the config.yaml to define the hyperparameters for the model training:
    # python src/mlops_project/model.py --config-path configs/config.yaml
    #
    # Also possible to define the hyperparameters directly:
    # python src/mlops_project/model.py --lr 0.001 --batch-size 32 --epochs 10
    #
    # hyperparameters options:
    #   batch_size
    #   epochs
    #   Learning rate = lr
    
    # Check if the config_path is not None
    if config_path != None:
        # Check if the config_path exists
        if Path(config_path).exists():
            # Load the specified .yaml configuration file
            config = OmegaConf.load(config_path)
            
            # Access the hyperparameters from the loaded config
            lr = config.hyperparameters.lr
            batch_size = config.hyperparameters.batch_size
            epochs = config.hyperparameters.epochs
            
            print(f"Loading configuration from {config_path}")
    
    # Print the hyperparameters
    print(f"Learning rate: {lr}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
            
    # Load the model
    bert = AutoModel.from_pretrained("bert-base-uncased", return_dict=False)
    model = Model(bert, lr)
    print(f"Model architecture: {model}")
    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Callbacks
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
    )

    # Initialize the trainer
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        max_epochs=epochs,
        limit_train_batches=0.2,
        callbacks=[early_stopping_callback, checkpoint_callback],
        logger=pl.loggers.WandbLogger(project="dtu_mlops"),
    )

    # Train the model
    trainer.fit(
        model,
        train_dataloaders=model.train_dataloader(batch_size),
        val_dataloaders=model.val_dataloader(batch_size),
    )

    # Save the model and log it to wandb
    torch.save(model.state_dict(), "Models/model.pth")


if __name__ == "__main__":
    typer.run(train)
