import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim,output_dim)
    def forward(self,x):
        pred_y = self.linear(x)
        return pred_y
    
class FFNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FFNN,self).__init__()
        # Define the network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, 600),
            nn.ReLU(),
            nn.Linear(600, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        # Define the forward pass
        print("Forward pass: ", x.shape)
        return self.network(x)
    