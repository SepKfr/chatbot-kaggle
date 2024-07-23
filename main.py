import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset

df = pd.read_csv('train.csv')


class ChatbotDataset(Dataset):

    def __init__(self, data, tokenizer, max_length=384):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        prompt = item['prompt']
        response_A = item['response_a']
        response_B = item['response_b']

        label = 0 if item['winner_model_a'] else 1 if item['winner_model_b'] else 2
        # Tokenize inputs
        inputs = self.tokenizer(prompt, response_A, response_B,
                                padding='max_length', max_length=self.max_length,
                                truncation=True, return_tensors='pt')

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label)
        }


tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large", num_labels=3)

# Unfreeze the specified parameters
for name, param in model.named_parameters():
    if name in ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Prepare dataset and dataloaders
train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)
train_dataset = ChatbotDataset(train_df, tokenizer)
val_dataset = ChatbotDataset(val_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):

    model.train()

    print("training....")

    for batch in train_loader:

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0
    total = 0
    correct = 0

    with torch.no_grad():

        for batch in val_loader:

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            print(outputs.logits)
            val_loss += outputs.loss.item()

            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {loss.item():.4f}, '
          f'Val Loss: {val_loss/len(val_loader):.4f}, '
          f'Val Acc: {(correct/total)*100:.2f}%')