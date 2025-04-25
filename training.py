import torch
import torch.nn as nn
from model import LinearRegression
from model import FFNN
from helpers import create_dataloaders


def train_model(train_loader, input_dim, output_dim, learning_rate, epochs):
    model = FFNN(input_dim,output_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            #print("Batch Embeddings Shape:", batch["embeddings"].shape)
            pred_y = model(batch["embeddings"].squeeze(1))
            loss = criterion(pred_y, batch["memory_used"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch} Loss: {epoch_loss}")
    
    torch.save(model.state_dict(), "/Users/dn10/Downloads/Bsub_dataset/model.pth")
    return model