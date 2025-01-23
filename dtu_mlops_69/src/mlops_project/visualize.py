import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import typer
from model import Model
from transformers import AutoModel
from data import tweets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


def visualize(model_checkpoint: str = "models/model.pth", figure_name: str = "confusion_matrix.png") -> None:
    """Visualize model predictions."""
    bert = AutoModel.from_pretrained('bert-base-uncased', return_dict=False)
    model = Model(bert, lr=0.001)

    # Load the model weights
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()  # Set the model to evaluation mode

    _, val_set = tweets()
    val_dataloader = DataLoader(val_set, batch_size=32, num_workers=4)

    # predictions and true labels
    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch in val_dataloader:
            sent_id, mask, targets = batch
            outputs = model(sent_id, mask)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds)
            all_true.append(targets)

    # Convert lists to tensors
    all_preds = torch.cat(all_preds)
    all_true = torch.cat(all_true)

    accuracy = accuracy_score(all_true.cpu(), all_preds.cpu())
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # Generate confusion matrix
    cm = confusion_matrix(all_true.cpu(), all_preds.cpu())

    # Plot confusion matrix using seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
                'Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f"reports/figures/{figure_name}")
    plt.show()


if __name__ == "__main__":
    typer.run(visualize)
