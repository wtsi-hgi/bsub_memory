import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

#### Mean Pooling for BERT Embeddings####
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return torch.sum(token_embeddings * input_mask_expanded, 1)/torch.clamp(input_mask_expanded.sum(1), min=1e-9)

####Load the model and tokenizer####

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-TinyBERT-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-TinyBERT-L6-v2')

####Get Embeddings####
def get_embeddings(command):
    encoded_input = tokenizer(command, padding=True, truncation= True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    return embeddings

####Preprocessing and Downsampling####
def preprocess_df(df, column= "MAX_MEM_USAGE_MB", min_mem_mb = 1.0, quantile = 0.99, bins =100, samples_per_bin=1000, random_state=42):
    # Filter jobs with low memory
    df = df[df[column] >= min_mem_mb].copy()
    # Calculate the upper bound using quantile
    upper_bound = df[column].quantile(quantile)
    
    # Further filter rows based on the upper bound
    df = df[df[column] <= upper_bound].reset_index(drop=True)
    
    # Bin the data
    #df['bin'] = pd.qcut(df[column], q=bins, duplicates='drop')
    df['bin'] = pd.cut(df[column], bins=bins, duplicates='drop')
    
    # Sample from each bin
    sampled_df = df.groupby('bin').apply(lambda x: x.sample(min(len(x), samples_per_bin), random_state=random_state))
    
    # drop the bin column
    sampled_df = sampled_df.reset_index(drop=True)
    sampled_df = sampled_df.drop(columns=['bin'])
    # Reset index
    sampled_df = sampled_df.reset_index(drop=True)
    
    return sampled_df

#####Loading the data#####
def load_json_file(input_path, output_path, keyword = "#BSUB", size_limit_gb = 5):
    """
    Processesing a large gzip JSONL file in chunks, filters for rows containing `keyword`, and writes until `size_limit_gb`.
    """
    size_limit = size_limit_gb * 1024 * 1024 * 1024  # Convert to bytes

    with open(output_path, "w") as output_file:
        for chunk in pd.read_json(input_path, lines=True, compression='gzip', chunksize=100000):
            filtered_chunk = chunk[chunk['Command'].str.contains(keyword, case=False, na=False)]
            filtered_chunk.to_json(output_file, orient='records', lines=True)

            if os.path.getsize(output_path) >= size_limit:
                print("File size limit reached. Stopping.")
                break


### Custom Dataset Class####
class CommandDataset(Dataset):
    def __init__(self, df):

        self.commands = df["Command"]
        self.memory_requested = torch.tensor(df["MEM_REQUESTED_MB"].values, dtype=torch.float32)
        self.memory_used = torch.tensor(df["MAX_MEM_USAGE_MB"].values, dtype=torch.float32)
        self.num_of_processors = torch.tensor(df["NUM_EXEC_PROCS"].values,dtype=torch.float32)
        self.embeddings = [torch.tensor(embedding, dtype=torch.float32) for embedding in df["Embeddings"]]
        
        print(f"Initialized CommandDataset with {len(self)}")
    
    def __len__(self):
        return len(self.commands)

    
    def __getitem__(self, idx):
            commands = self.commands.iloc[idx]
            
            memory_requested = self.memory_requested[idx]
            memory_used = self.memory_used[idx]
            num_of_processors = self.num_of_processors[idx]
            embeddings = self.embeddings[idx]

            data = {
                "command": commands,
                "memory_requested": memory_requested,
                "memory_used": memory_used,
                "num_of_processors": num_of_processors,
                "embeddings": embeddings
            }

            return data


### Create dataloader####
def create_dataloaders(json_path: str, batch_size: int = 32, test_size: float = 0.2, min_mem_mb: float = 1.0, 
                       quantile: float = 0.99, 
                       bins: int = 100, 
                       samples_per_bin: int = 1000, 
                       random_state: int = 42) -> tuple:
    df = pd.read_json(json_path, lines= True)
    df = preprocess_df(df, column="MAX_MEM_USAGE_MB", min_mem_mb=1.0, quantile=0.99, bins=100, samples_per_bin=1000, random_state=42)
    #print(df.columns)
    #print(df.iloc[0])
    df["Embeddings"] = df["Command"]. apply(lambda x : get_embeddings(x))
    data = CommandDataset(df)
    data.embeddings = [embedding.squeeze(1) for embedding in data.embeddings]
    
    train_size = int(len(data) * (1 - test_size))
    test_size = len(data) - train_size
    train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])
    print(len(train_data))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
        

    