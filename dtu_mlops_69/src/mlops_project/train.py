"""
We don't need this if we use the Pytorch Lightning framework?

import matplotlib.pyplot as plt
import torch
import typer
from model import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    model = Model().to(DEVICE)
    
    # Load data
    #train_set, _ = ?
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    
    # Loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # statistics
    statistics = {"train_loss": [], "train_accuracy": []}
    
    for epoch in range(epochs):
        model.train()

        for i, (?, ?) in enumerate(train_dataloader):
            optimizer.zero_grad()
            y_pred = model(?)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
             

        
    # Save model
    #torch.save(model.state_dict(), "models/model.pth")
 """