from helpers import create_dataloaders, load_json_file
from model import LinearRegression
from training import train_model
from inference import evaluate_model, run_inference
import matplotlib.pyplot as plt
import torch

def main():
    #Define the path and parameters#
    input_path ="/Users/dn10/Downloads/Bsub_dataset/data.jsonl.gz"
    output_path ="/Users/dn10/Downloads/Bsub_dataset/filtered_under_5GB.jsonl"
    epochs = 20
    learning_rate = 0.00001

    #Create dataloaders#
    print(f'Loading data and creating dataloaders')
    load_json_file(input_path=input_path, output_path=output_path, keyword = "#BSUB", size_limit_gb = 5)
    json_path = output_path
    print(f'Loading data from {json_path}')
    train_loader, test_loader = create_dataloaders(json_path=json_path, batch_size=32, test_size=0.2, min_mem_mb=1.0,quantile=0.99,bins=100,samples_per_bin=1000,random_state=42)

    #Define input and output dimensions#
    print(next(iter(train_loader)))
    first_batch = next(iter(train_loader))
    first_batch["embeddings"] = first_batch["embeddings"].squeeze(1)
    input_dim = first_batch["embeddings"].shape[1]
    print(f'Input dimension: {input_dim}')
    output_dim = 1
    print(first_batch["embeddings"].shape)

    #Initialize the model and train the model#
    print(f'Training the model')
    model = train_model(train_loader, input_dim = input_dim, output_dim = output_dim, learning_rate = learning_rate, epochs = epochs)

    # Load the saved model
    model.load_state_dict(torch.load('/Users/dn10/Downloads/Bsub_dataset/model.pth'))

    print(f'Evaluate the model')
    total_predictions, total_actual = evaluate_model(test_loader, model)

    #Plot the predictions and actual values#
    plt.scatter(total_actual, total_predictions)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.show()

if __name__ == "__main__":
    main()



