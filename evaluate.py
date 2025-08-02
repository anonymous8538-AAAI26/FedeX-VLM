import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
import pandas as pd
from transformers import ViTModel,TFBertModel
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from transformers import BertTokenizer, BertModel,BertConfig
import torch
import os
from transformers import ViTModel, ViTFeatureExtractor
from torchvision import transforms, models
import time

labelencoder = LabelEncoder()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

bert_size=128

model_method='vit_bert_all_concat_bert_transformer'


if 'swin' in model_method:
    from transformers import AutoImageProcessor
    from transformers import SwinModel,T5Model
    print('swin')
    
dataset_type='VQA_v1'    #VQA_v1 /VQA_v2

if  'random' in dataset_type:
    title_mid='random'
else:
    title_mid='all'
   

folder ='WeightedVQA_v1FEDUlen_clientall15_vit_bert_all_concat_bert_transformerepcoh_50soft_max'     #saved name of model

csv_folder=f'Datasets/{dataset_type}/'

labelencoder_df=pd.read_csv(csv_folder+f"Train_{dataset_type}_preprocessed.csv")
labelencoder.fit(labelencoder_df['answer_preprocessed'])
num_classes = len(labelencoder_df['answer_preprocessed'].unique())


x_test=pd.read_csv(csv_folder+f"/X_Val_{dataset_type}_preprocessed.csv")

x_test['answer_labelencoded']=labelencoder.transform(x_test['answer_preprocessed'])
y_test=x_test['answer_labelencoded']




if model_method=='swinB_bert_all_concat_bert_transformer':
    vit_feature_extractor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
else:
    vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])  




if model_method=='vit_bert_all_concat_bert_transformer':
    class ViTBertConcatTransformer(nn.Module):
        def __init__(self, vit_model_name='google/vit-base-patch16-224-in21k', bert_model_name='bert-base-uncased', num_classes=num_classes):
            super(ViTBertConcatTransformer, self).__init__()
    
            self.vit = ViTModel.from_pretrained(vit_model_name)
            self.bert = BertModel.from_pretrained(bert_model_name)
    
            
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
      
            combined_embeddings = torch.cat((vit_concat, bert_concat), dim=1)
            #print('combined_embeddings', combined_embeddings.shape)
    
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
        
        
elif model_method=='vit_Bt5_all_concat_bert_transformer':
    class ViTBertConcatTransformer(nn.Module):
        def __init__(self, vit_model_name='google/vit-base-patch16-224-in21k', t5_model='t5-base',bert_model_name='bert-base-uncased', num_classes=num_classes):
            super(ViTBertConcatTransformer, self).__init__()

          
            self.vit = ViTModel.from_pretrained(vit_model_name)
            self.bert = T5Model.from_pretrained(t5_model)

            full_bert_model = BertModel.from_pretrained(bert_model_name)
          
            self.bert_encoder  = full_bert_model.encoder

            # Fully Connected Layer for classification
            self.fc = nn.Linear(full_bert_model.config.hidden_size, num_classes)
            
            
            
        def forward(self, image, text):
            # ViT forward pass
            
            vit_outputs = self.vit(pixel_values=image)
            vit_concat = vit_outputs.last_hidden_state  # CLS token + Patch tokens
 
            bert_outputs = self.bert.encoder(input_ids=text['input_ids'], attention_mask=text['attention_mask'])
            bert_concat = bert_outputs.last_hidden_state  # CLS token + Token embeddings
  
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
            #print('combined_embeddings', combined_embeddings.shape)
    
            # Pass through BERT encoder
            encoder_outputs = self.bert_encoder(combined_embeddings)
            transformer_output = encoder_outputs.last_hidden_state
    
            # Classification output
            output = self.fc(transformer_output[:, 0, :])  # Use the CLS token for classification
    
            return output
            
    
from PIL import Image          

def calculate_accuracy(predictions, x_test, y_test):
  
    predicted_classes = labelencoder.inverse_transform(predictions)

    acc_val_lst = []
 
    actual_answers_list = x_test['answer_list_preprocessed'].apply(lambda ans: ans.strip("[]").replace("'", "").split(", "))

    for i, predicted_class in enumerate(predicted_classes):
        
        actual_answers = [ans.strip() for ans in actual_answers_list[i]]

    
        match_count = sum(1 for ans in actual_answers if ans == predicted_class)

      
        acc_val = 1 if match_count >= 3 else match_count / 3.0
        acc_val_lst.append(acc_val)


    test_accuracy = (sum(acc_val_lst) / len(y_test)) * 100
    print('-test_accuracy', test_accuracy)

    return test_accuracy
    


    
    
from torch.utils.data import DataLoader, Dataset
class CustomDataset(Dataset):
    def __init__(self, x_data, y_data, tokenizer, feature_extractor, transform=None):
        self.x_data = x_data
        self.y_data = y_data
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
      
        if dataset_type=='textvqa':
            image_id = self.x_data.iloc[idx]['image_id']
            new_image_id = f"textvqa_train_images/{image_id}.jpg"

          
            image_id = self.x_data.at[idx, 'image_id'] = new_image_id
            #print('image_id',image_id)

        else:
        
            image_id = self.x_data.iloc[idx]['image_id']
        answers = self.x_data.iloc[idx]['answer_list_preprocessed']
        image_path = os.path.join(image_id)
        #image_path = os.path.join(base_path, image_id)
        
        image = Image.open(image_path).convert('RGB')
        
      
        if self.transform:
            image = self.transform(image)
        else:
            # VitF
            image = self.feature_extractor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        
        text = self.x_data.iloc[idx]['question_preprocessed']
        text_inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
        #print('text',text)
       
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
        
     
        label = torch.tensor(self.y_data[idx], dtype=torch.long).squeeze()  # Ensure label is a 1D tensor

        return {'image': image, 'text': text_inputs, 'label': label,'answers':answers}
        

    
def accuracy_measure_fed(model,all_predictions,client_index_model):

    print('all_predictionstesttest',len(all_predictions))
    with torch.no_grad(): 
        for idx,batch in enumerate(test_dataloader):
            start_time = time.time() 
            print('batch',idx , '/',len(test_dataloader))
    
            images = batch['image'].to(device)
            texts = {k: v.to(device) for k, v in batch['text'].items()}
        
            outputs = model(images, texts)
            _, predicted = torch.max(outputs, 1)
    
            all_predictions.append(predicted.cpu().numpy())
            torch.cuda.empty_cache()
            end_time = time.time()  
            elapsed_time = end_time - start_time  
            print(f"{client_index_model} Time for batch {idx + 1}/{len(test_dataloader)}: {elapsed_time:.4f} seconds")
            
            
    all_predictions = np.concatenate(all_predictions)
   
    # Calculate accuracy using the labels and predictions
    accuracy = calculate_accuracy(all_predictions, x_test, y_test)   
    accuracy_path = 'Accuracy.txt'
    with open(accuracy_path,"a") as file:
        file.write(client_index_model+'\n'+str(accuracy)+'\n\n')
        
    return accuracy
                
            
            
                
    
test_dataset = CustomDataset(x_test, y_test, bert_tokenizer, vit_feature_extractor, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False,num_workers=4, pin_memory=True)


model_files = [f for f in os.listdir(folder) if f.endswith('.pt')]
model_name = [os.path.join(folder, f) for f in model_files]

for index,client_index_model in enumerate(model_name):

    all_predictions = []
    model = ViTBertConcatTransformer()
    
    model.load_state_dict(torch.load(client_index_model, map_location=device))
    
    model.to(device)
    
    model.eval()    
    accuracy=accuracy_measure_fed(model,all_predictions,client_index_model)
   
