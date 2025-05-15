from torch.utils.data import DataLoader
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from DatasetCreator import Dataset  # Your custom dataset class
from ClipBertViewPredictor import ClipBertViewPredictor  # Your model class
from YouTubeDataset import YouTubeDataset


def predict_and_save(model, test_dataset, batch_size=16):

    
    print("Testing model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)

    # Custom collate function
    def collate_fn(batch):
        return {
            "image": torch.stack([x["image"] for x in batch]),
            "title": [x["title"] for x in batch],
            "description": [x["description"] for x in batch],
            "tabular": torch.stack([x["tabular"] for x in batch]),
            "id": [x["id"] for x in batch]
        }
    predictions=[]
    ids=[]
    print("Creating DataLoader...")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)    
    progress_bar = tqdm(test_loader,desc="Testing", leave=False)
    for batch in progress_bar:
        # Move to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items() if k != "target"}
        outputs = model(inputs).view(-1).cpu()  # Move to CPU immediately
        predictions.extend(outputs.detach().numpy())  # Store as numpy array
        ids.extend(batch["id"])
    df = pd.DataFrame({
        'ID': ids,  # Or any identifier column
        'views': predictions  # Match competition requirements
    })
    df["views"]=np.exp(df["views"])
    assert len(df) == 3402, f"Expected 3402 rows, got {len(df)}"
    
    df.to_csv("./artifacts/results.csv", index=False)
    print(f"Submission saved")


if __name__ == '__main__':
    # Train and test
    ratio = 1
    test_data = YouTubeDataset("./dataset/processed_test_set.csv", "./dataset/test/", training = False)
    model = ClipBertViewPredictor()  # Replace with your model class and arguments

    # Step 2: Load the saved state_dict
    model.load_state_dict(torch.load('./models/best_model_clipbertpredictor.pth'))

    # Step 3 (optional but recommended): Set model to evaluation mode
    model.eval()
        

    predict_and_save(model, test_data)
    