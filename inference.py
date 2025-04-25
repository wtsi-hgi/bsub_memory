import torch 
import torch.nn as nn
import numpy
from helpers import create_dataloaders


criterion = nn.MSELoss() 
def evaluate_model(test_loader,model):
    model.eval()
    test_loss = 0
    total_predictions = numpy.array([])
    total_actual = numpy.array([])
    with torch.no_grad():
        for batch in test_loader:
            pred_y = model(batch["embeddings"])
            loss = criterion(pred_y, batch["memory_used"])
            test_loss += loss.item()
            print(f"Predicted: {pred_y}, Actual: {batch['memory_used']}")
            total_predictions = numpy.append(total_predictions,pred_y.numpy())
            total_actual = numpy.append(total_actual,batch["memory_used"].numpy())
    
    test_loss = test_loss/len(test_loader)
    print(f"Test Loss: {test_loss}")
    
    return total_predictions, total_actual

###Run inference on a new input data###
def run_inference(model, input_data):
    model.eval()
    with torch.no_grad():
        pred_y = model(input_data)
    
    return pred_y
        
   

    
        
   