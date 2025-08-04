import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from transformers import SwinModel,BertTokenizer, BertModel,BertConfig
from transformers import BertTokenizer, BertModel,BertConfig
from transformers import AutoImageProcessor
from datetime import datetime
import re
import csv
import copy
import time
import glob
import random
from torch.optim import Adam
from transformers import ViTModel, ViTFeatureExtractor
import torch.nn.functional as F
import statistics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

# Load hyperparameters and configuration from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

learning_rate = config['learning_rate']
epsilon = config['epsilon']
batch_size = config['batch_size']
round_num = config['round_num']
alpha = config['alpha']
dataset_type = config['dataset']
num_clients=config['num_clients']
normalize=config['normalize']


# Set device 
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)

# Define model architecture identifier
model_method='vit_bert_all_concat_bert_transformer'


# Federated learning settings
num_epochs_per_round = 1
start_epoch=0
soft_max=1

# Determine data splitting method
if 'random' in dataset_type:
    datasplit_type='random'
else:
    datasplit_type='all'

# Set path to dataset CSVs
csv_folder=f'DATASETs/{dataset_type}/'
# If softmax is used, add tag for naming
if soft_max==1:
    soft_max_title='soft_max'


save_dir =f"saved_model/{normalize}_alpha{alpha}_Weighted{dataset_type}FED{datasplit_type}{num_clients}{model_method}epcoh_{round_num}{soft_max_title}"
    
      

     
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    

labelencoder = LabelEncoder()
labelencoder_df=pd.read_csv(csv_folder+f"Train_{dataset_type}_preprocessed.csv")
labelencoder.fit(labelencoder_df['answer_preprocessed'])
num_classes = len(labelencoder_df['answer_preprocessed'].unique())

    
    
    
def load_test_data(csv_folder):
    x_test=pd.read_csv(csv_folder+"10000_x_test.csv")
    x_test['answer_labelencoded']=labelencoder.transform(x_test['answer_preprocessed'])
    y_test=x_test['answer_labelencoded']
    
    return x_test, y_test



def load_and_split_data(num_clients):
    
    x_train_clients, y_train_clients = [], []
    

    if 'all' in datasplit_type:
        for i in range(num_clients):

            x_train = pd.read_csv(f"{csv_folder}/{num_clients}clients/X_{i+1}.csv")
            x_train['answer_labelencoded']=labelencoder.transform(x_train['answer_preprocessed'])
            y_train=x_train['answer_labelencoded']
        
          
            
            x_train_clients.append(x_train)
            y_train_clients.append(y_train)
    
    elif 'random' in datasplit_type:
        for i in range(num_clients):
            
            x_train=pd.read_csv(f"{csv_folder}/{num_clients}clients/X_train_index{i+1}.csv")
            x_train['answer_labelencoded']=labelencoder.transform(x_train['answer_preprocessed'])
            y_train=x_train['answer_labelencoded']
            x_train_clients.append(x_train)
            y_train_clients.append(y_train)


    return x_train_clients, y_train_clients




x_train_clients, y_train_clients = load_and_split_data(num_clients)

    
class CustomDataset(Dataset):
    def __init__(self, x_data, y_data, tokenizer, feature_extractor, transform=None):
        self.x_data = x_data.reset_index(drop=True)  # Ensure index is reset
        self.y_data = y_data.reset_index(drop=True)  # Ensure index is reset
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        idx = int(idx)  # Ensure idx is an integer
        
        if dataset_type=='textvqa':
            image_id = self.x_data.iloc[idx]['image_id']
            new_image_id = f"textvqa_train_images/{image_id}.jpg"

          
            image_id = self.x_data.at[idx, 'image_id'] = new_image_id
            #print('image_id',image_id)

        else:
        
          image_id = self.x_data.iloc[idx]['image_id']
          
          
          
        answers = self.x_data.iloc[idx]['answers']
        image_path = os.path.join(image_id)
        image = Image.open(image_path).convert('RGB')
        
        

        if self.transform:
            image = self.transform(image)
        else:
            image = self.feature_extractor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

        text = self.x_data.iloc[idx]['question_preprocessed']
        text_inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
      
        
        label = torch.tensor(self.y_data.iloc[idx], dtype=torch.long).squeeze()  # Ensure label is a 1D tensor
        
        return {'image': image, 'text': text_inputs, 'label': label, 'answers': answers}
        
        
                                    
if model_method=='vit_bert_all_concat_bert_transformer':
    class ViTBertConcatTransformer(nn.Module):
        def __init__(self, vit_model_name='google/vit-base-patch16-224-in21k', bert_model_name='bert-base-uncased', num_classes=num_classes):
            super(ViTBertConcatTransformer, self).__init__()

            # Load pre-trained Vision Transformer (ViT) model
            self.vit = ViTModel.from_pretrained(vit_model_name)
            # Load pre-trained BERT model
            self.bert = BertModel.from_pretrained(bert_model_name)

            full_bert_model = BertModel.from_pretrained(bert_model_name)
     
            self.bert_encoder  = full_bert_model.encoder

            # Fully Connected Layer for classification
            self.fc = nn.Linear(full_bert_model.config.hidden_size, num_classes)
            
            
            
        def forward(self, image, text):
            # ViT forward pass
            vit_outputs = self.vit(pixel_values=image)
            vit_concat = vit_outputs.last_hidden_state  # CLS token + Patch tokens
            # Forward pass through BERT
            bert_outputs = self.bert(input_ids=text['input_ids'], attention_mask=text['attention_mask'])
            bert_concat = bert_outputs.last_hidden_state  # CLS token + Token embeddings
            # Concatenate ViT and BERT outputs along the sequence dimension
            combined_embeddings = torch.cat((vit_concat, bert_concat), dim=1)
            # Pass the combined sequence through the BERT encoder
            encoder_outputs = self.bert_encoder(combined_embeddings)
            transformer_output = encoder_outputs.last_hidden_state

            # Classification output
            output = self.fc(transformer_output[:, 0, :])  # Use the CLS token for classification
                
            return output
                       
elif model_method=='Lvit_bert_all_concat_bert_transformer':
    class ViTBertConcatTransformer(nn.Module):
        def __init__(self, vit_model_name='google/vit-large-patch16-224-in21k', bert_model_name='bert-base-uncased', num_classes=num_classes):
            super(ViTBertConcatTransformer, self).__init__()

           
            self.vit = ViTModel.from_pretrained(vit_model_name)
            self.bert = BertModel.from_pretrained(bert_model_name)
            
            self.vit_projector = nn.Linear(self.vit.config.hidden_size, 768) 
            
            full_bert_model = BertModel.from_pretrained(bert_model_name)
     
            self.bert_encoder  = full_bert_model.encoder

            # Fully Connected Layer for classification
            self.fc = nn.Linear(full_bert_model.config.hidden_size, num_classes)
            
            
            
        def forward(self, image, text):
            # ViT forward pass
            
            vit_outputs = self.vit(pixel_values=image)
            vit_concat = vit_outputs.last_hidden_state  # CLS token + Patch tokens
 
            bert_outputs = self.bert(input_ids=text['input_ids'], attention_mask=text['attention_mask'])
            bert_concat = bert_outputs.last_hidden_state  # CLS token + Token embeddings
            vit_concat = self.vit_projector(vit_concat)  # (batch_size, seq_len_vit, 768)

            
            
            
            combined_embeddings = torch.cat((vit_concat, bert_concat), dim=1)
            #print('combined_embeddings', combined_embeddings.shape)
    
            encoder_outputs = self.bert_encoder(combined_embeddings)
            transformer_output = encoder_outputs.last_hidden_state

            # Classification output
            output = self.fc(transformer_output[:, 0, :])  # Use the CLS token for classification
                
            return output
        
        

elif model_method=='swinB_bert_all_concat_bert_transformer':
    class ViTBertConcatTransformer(nn.Module):
        def __init__(self, swin_model_name='microsoft/swin-base-patch4-window7-224', bert_model_name='bert-base-uncased', num_classes=num_classes):
            super(ViTBertConcatTransformer, self).__init__()
    
            # Load Swin-B encoder
            self.swin = SwinModel.from_pretrained(swin_model_name)
    
            # Load BERT encoder
            self.bert = BertModel.from_pretrained(bert_model_name)
            full_bert_model = BertModel.from_pretrained(bert_model_name)
            self.bert_encoder = full_bert_model.encoder
            self.swin_transform = nn.Linear(self.swin.config.hidden_size, full_bert_model.config.hidden_size)

            # Fully Connected Layer for classification
            self.fc = nn.Linear(full_bert_model.config.hidden_size, num_classes)
    
        def forward(self, image, text):
            # Swin-B forward pass
            swin_outputs = self.swin(pixel_values=image)
            swin_concat = swin_outputs.last_hidden_state  # Patch tokens
            swin_concat = self.swin_transform(swin_concat) 

            # BERT forward pass
            bert_outputs = self.bert(input_ids=text['input_ids'], attention_mask=text['attention_mask'])
            bert_concat = bert_outputs.last_hidden_state  # CLS token + Token embeddings
    
            # Concatenate embeddings from Swin-B and BERT
            combined_embeddings = torch.cat((swin_concat, bert_concat), dim=1)
         
            # Pass through BERT encoder
            encoder_outputs = self.bert_encoder(combined_embeddings)
            transformer_output = encoder_outputs.last_hidden_state
    
            # Classification output
            output = self.fc(transformer_output[:, 0, :])  # Use the CLS token for classification
    
            return output
            
if model_method=='swinB_bert_all_concat_bert_transformer':
    vit_feature_extractor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
else:
    vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def create_data_loaders(x_train, y_train, tokenizer, feature_extractor, batch_size=batch_size):
    train_dataset = CustomDataset(x_train, y_train, tokenizer, feature_extractor, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    
    return train_dataloader
    
def create_test_dataloader(x_test, y_test, tokenizer, feature_extractor, batch_size=batch_size):
    test_dataset = CustomDataset(x_test, y_test, tokenizer, feature_extractor, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return test_dataloader
        
 
train_dataloaders = []
val_dataloaders = []


for i in range(num_clients):
    train_dl = create_data_loaders(x_train_clients[i], y_train_clients[i], bert_tokenizer, vit_feature_extractor)
    train_dataloaders.append(train_dl)


x_test, y_test = load_test_data(csv_folder)
test_dataloader = create_test_dataloader(x_test, y_test, bert_tokenizer, vit_feature_extractor, batch_size=batch_size)
    
                   
def train_local_model(model, dataloader, optimizer, criterion, client_idx,round_value,num_epochs):
    model.train()
    
  
    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0.0
        correct_preds = 0
        total_samples = 0

    
        # Use tqdm to display progress bar
        dataloader_tqdm = tqdm(dataloader, desc=f"Client {client_idx + 1} Epoch {epoch + 1}/{num_epochs}", unit='batch')
        
        for batch in dataloader_tqdm:
            images = batch['image'].to(device)
            texts = {k: v.to(device) for k, v in batch['text'].items()}
            labels = batch['label'].to(device).squeeze()
            if labels.dim()==0:
                labels=labels.unsqueeze(0) 
            optimizer.zero_grad()
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            train_accuracy = correct_preds / total_samples
            # Update tqdm description with current training loss
            dataloader_tqdm.set_postfix({'train_loss': total_loss / total_samples,'train_accuracy':train_accuracy})


        avg_loss = total_loss / total_samples
        train_acc = correct_preds / total_samples
        epoch_duration = time.time() - start_time
        
        
        
        test_accuracy = evaluate_test(model, test_dataloader, x_test, y_test, labelencoder)
        print(f"client{client_idx + 1} Model Test Accuracy: {test_accuracy:.2f}%")
        
        
        # Save metrics
        save_metrics(client_idx+1 , round_value + 1, avg_loss, train_acc,test_accuracy, epoch_duration,0)
        
       
        model_filename = f"client{client_idx+ 1}_round_{round_value+1}_epoch_{epoch + 1}_test_accuracy_{test_accuracy:.4f}.pt"
        model_path = os.path.join(save_dir, model_filename)
        torch.save(model.state_dict(), model_path)
        print(f"Saved best model at {model_path}")
        
           
    return model,avg_loss


                           

    
def fedex(global_model, client_models, weights,client_loss_list):
    """
    Aggregates local client models into a global model using weighted averaging,
    considering both knowledge level weights and their local training losses.
    
    Args:
        global_model (nn.Module): The current global model to be updated.
        client_models (list of nn.Module): List of trained client models.
        weights (list of float): Predefined client-specific weights (e.g., based on data heterogeneity).
        client_loss_list (list of float): Local training losses from each client.
    
    Returns:
        global_model (nn.Module): The updated global model after aggregation.
    """

    # Get the state dictionary (parameters) of the global model
    global_state_dict = global_model.state_dict()
    num_clients = len(client_models)
    
    # Normalize the input weights and client losses (if specified)
    if normalize == 'minmax':
        # Min-max normalization for weights
        weight_min_val = min(weights)
        weight_max_val = max(weights)
        weights = [(x - weight_min_val) / (weight_max_val - weight_min_val) for x in weights]
        loss_min=min(client_loss_list)
        loss_max=max(client_loss_list)
        client_loss_list=[(x - loss_min) / (loss_max - loss_min) for x in client_loss_list]

        
    elif normalize=='z_score':
        # Z-score normalization for weights
        mean_val = statistics.mean(weights)
        std_val = statistics.stdev(weights)
        weights = [(x - mean_val) / (std_val+epsilon) for x in weights]
        # Z-score normalization for client losses
        loss_mean_val = statistics.mean(client_loss_list)
        loss_std_val = statistics.stdev(client_loss_list)
        client_loss_list = [(x - loss_mean_val) / loss_std_val for x in client_loss_list]

    # Compute final weights using softmax 
    if soft_max==1:
        client_loss_list = torch.tensor(client_loss_list, dtype=torch.float32)
        inv_loss = 1.0 / (client_loss_list + epsilon) 
        
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        
        final_weights = alpha * weights_tensor + (1.0 - alpha) * inv_loss
        final_weights = F.softmax(torch.tensor(final_weights, dtype=torch.float32), dim=0)
        
        
    else:
        client_loss_list = torch.tensor(client_loss_list, dtype=torch.float32)
        inv_loss = 1.0 / (client_loss_list + epsilon) 
        
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        final_weights = alpha * weights_tensor + (1.0 - alpha) * inv_loss
        

    total_weight = final_weights.sum()
    # Weighted aggregation of model parameters from all clients
    for key in global_state_dict.keys():
        # Stack and weigh updates
        weighted_updates = torch.stack(
            [client_models[i].state_dict()[key].float() * final_weights[i] for i in range(num_clients)],
            dim=0
        )
        # Aggregate weighted updates
        global_state_dict[key] = weighted_updates.sum(dim=0) / total_weight

        
    # Load updated parameters into global model
    global_model.load_state_dict(global_state_dict)

    return global_model
    
        
                


   
def federated_learning(global_model, num_clients, num_epochs_per_round, round_num,start_epoch):
    """
    Performs federated learning across multiple clients for a specified number of communication rounds.

    Args:
        global_model (nn.Module): The initial global model to be distributed and aggregated.
        num_clients (int): Number of clients participating in federated learning.
        num_epochs_per_round (int): Number of local training epochs per round.
        round_num (int): Total number of federated learning rounds.
        start_epoch (int): Starting round (used for resuming training).

    Returns:
        global_model (nn.Module): The updated global model after all communication rounds.
    """   
    for round_value in range(start_epoch,round_num):
        print(f"Round {round_value + 1}/{round_num}")

        client_models = []
        client_loss_list=[]
        for client_idx in range(num_clients):
            # Initialize a local model and optimizer
            client_model = copy.deepcopy(global_model)
            optimizer = Adam(client_model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # Retrieve the dataloaders for the current client
            train_dataloader = train_dataloaders[client_idx]
        
        
            # Train the local model     
            trained_model,client_loss = train_local_model(client_model, train_dataloader, optimizer, criterion , client_idx,round_value ,num_epochs_per_round)
            client_models.append(trained_model)
            client_loss_list.append(client_loss)
            
        
        if dataset_type=='VQA_v1':
            if num_clients==2:
                answer_hetero=[1,1.42]
            elif num_clients==3:
                answer_hetero=[1,1,1.64]
            elif num_clients==4:
                answer_hetero=[1,1,1,1.85]
            elif num_clients==5:
                answer_hetero=[1,1,1,1,2.06]
        elif dataset_type=='VQA_v2':
             
            if num_clients==2:
                answer_hetero=[1.42 ,2.56]
            elif num_clients==3:
                answer_hetero=[1.13, 2.0, 2.84]
            elif num_clients==4:
                answer_hetero=[ 1.0, 1.83, 2.0, 3.12]
            elif num_clients==5:
                answer_hetero=[ 1.0, 1.54 ,2.0, 2.0, 3.4]
            elif num_clients==10:
                answer_hetero=[1, 1, 1.09, 2, 2, 2, 2, 2, 2, 4.81]
            elif num_clients==15:
                answer_hetero=[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6.21]
            elif num_clients==20:
                answer_hetero=[1, 1, 1, 1, 1, 1.17, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2.32, 7.3]


        elif dataset_type=='VQA_v2_random':
            if num_clients==5:
                answer_hetero=[1.29,1.29,1.29,1.29,1.29]
            
            
        global_model=fedex(global_model,client_models,answer_hetero,client_loss_list)
    
        
        # Evaluate the global model
        test_accuracy = evaluate_global_model(global_model, test_dataloader, x_test, y_test, labelencoder, round_value+1)
        print(f"Global Model Test Accuracy: {test_accuracy:.2f}%")



    return global_model
    
    
        
    


def calculate_accuracy(predictions, x_test, y_test):
    """
    Calculates accuracy for VQA-style tasks.
    An answer is considered fully correct if it matches at least 3 human-labeled answers.
    
    Args:
        predictions (list): List of predicted class indices.
        x_test (DataFrame): Test features including human-annotated answers.
        y_test (list or array): Ground truth labels.
    
    Returns:
        float: Final test accuracy in percentage.
    """

    acc_val_lst = []

    for i, pred in enumerate(predictions):
       
        predicted_classes = labelencoder.inverse_transform([pred])[0]
        
   
        answers_str = list(x_test['answers'])[i].strip("[]")
        answers_list = answers_str.replace("'", "").split(", ")
        
        temp = 0
  
        for actual_ans in answers_list:
            actual_ans = actual_ans.strip() 
            if str(actual_ans) == predicted_classes:
                temp += 1
        
        if temp >= 3:
            acc_val = 1
        else:
            acc_val = float(temp) / 3

        acc_val_lst.append(acc_val)
    
    test_accuracy = (sum(acc_val_lst) / len(y_test)) * 100
    print('test_accuracy', test_accuracy)
    return test_accuracy
    


def save_metrics(client_idx, round_value, train_loss, train_acc, test_acc, duration, global_model=False):
    # Set the directory and file name based on whether it's a client or global model
    if global_model:
        file_name = 'global_model_metrics.csv'
    else:
        file_name = f'client_{client_idx}_metrics.csv'

    metrics_file = os.path.join(save_dir, file_name)
    file_exists = os.path.exists(metrics_file)
    
    
    
    with open(metrics_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            if global_model:
                # For global model, write header without loss
                writer.writerow(['Round', 'Test Accuracy'])
            else:
                # For client model, write header with all metrics
                writer.writerow(['Round', 'Train Loss', 'Train Accuracy', 'Test Accuracy', 'Time Taken (s)'])
        
        if global_model:
            # For global model, only write test accuracy
            writer.writerow([round_value, test_acc])
        else:
            # For client model, write all metrics
            writer.writerow([round_value, train_loss, train_acc, test_acc, duration])
            
            
            
            
        

def evaluate_test(model, dataloader, x_test, y_test, labelencoder):
    model.eval()
    correct_preds = 0
    total_samples = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            texts = {k: v.to(device) for k, v in batch['text'].items()}
            labels = batch['label'].to(device).squeeze()
            if labels.dim()==0:
                labels=labels.unsqueeze(0) 
            outputs = model(images, texts)
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            all_predictions.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)

    # Calculate accuracy using the labels and predictions
    test_accuracy = calculate_accuracy(all_predictions, x_test, y_test)

    return test_accuracy

    
def evaluate_global_model(global_model, test_dataloader, x_test, y_test, labelencoder, round_value):
    test_accuracy = evaluate_test(global_model, test_dataloader, x_test, y_test, labelencoder)
    save_metrics(client_idx=None, round_value=round_value, train_loss=None, train_acc=None, test_acc=test_accuracy, duration=None, global_model=True)
    print(f"Global Model Test Accuracy: {test_accuracy:.2f}%")    
    return test_accuracy
    
    


# Initialize the global model
global_model = ViTBertConcatTransformer()
global_model = global_model.to(device)



global_model = federated_learning(global_model, num_clients, num_epochs_per_round, round_num,start_epoch)
