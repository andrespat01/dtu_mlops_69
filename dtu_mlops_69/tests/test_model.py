import pytest
import torch
from transformers import AutoModel
from src.mlops_project.model import Model  # Replace with your actual model import

@pytest.mark.parametrize("batch_size", [32, 64])  # Add more batch sizes if needed
def test_model(batch_size: int) -> None:
    # Initialize the model
    bert = AutoModel.from_pretrained('bert-base-uncased', return_dict=False)
    model = Model(bert=bert, lr=0.001)
    
    # dummy values
    sent_id = torch.randint(0, 1000, (batch_size, 61))
    mask = torch.ones(batch_size, 61)
    targets = torch.randint(0, 2, (batch_size,))
    
    #forward pass
    outputs = model(sent_id, mask)
    assert outputs.shape == (batch_size, 2), "Model output shape is incorrect"
    assert isinstance(outputs, torch.Tensor), "Model output is not a tensor"
    
    batch = (sent_id, mask, targets)
    loss = model.training_step(batch, 0)
    assert loss is not None, "Loss is None"

