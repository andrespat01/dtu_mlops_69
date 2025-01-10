import torch
import typer
from model import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def evaluate(model_checkpoint: str = "models/model.pth") -> None:
    """Evaluate a trained model."""
    model = Model().to(DEVICE)
    
    # Load model checkpoint
    model.load_state_dict(torch.load(model_checkpoint))

    # Load test data
    #_, test_set = ?
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)
    
    model.eval()
    
    #....    

if __name__ == "__main__":
    typer.run(evaluate)