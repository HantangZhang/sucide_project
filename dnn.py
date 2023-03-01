import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# if using mac
df = pd.read_excel('/Users/zhanghantang/PycharmProjects/sucide_project/BIOM40forUSC.xlsx')
data = df[df['SI'].notnull()]
y = data['SI']
x = data.loc[:, 'GIMAP1Biom1552316_a_at':'CFIS']

# column 'CFI-S.PheneVisit' data type is string, i am not clear its internal meaning and how to convert to float data type
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
x = x.drop(labels='CFI-S.PheneVisit', axis=1)
# drop these column directly
x = data.loc[:, 'GIMAP1Biom1552316_a_at':'RAB3GAP2Biom240234_at']
new_y = []
for i in y:
    if i == 0 or i ==1:
        new_y.append(0)
    else:
        new_y.append(1)
y = np.array(new_y)
X = StandardScaler().fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, new_y, test_size=0.2, random_state=10)
print(len(X_train), len(X_test))

# todo unified randome state
unified_random_state = 32

# make sure every split have postive value

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.relu(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

# Define evaluation function
def evaluate(model, criterion, dataloader, device):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().view(-1, 1))
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    return epoch_loss, precision, recall, f1, roc_auc

# Training loop
def train(model, criterion, optimizer, train_dataloader, val_dataloader, device, num_epochs=10):
    # best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_dataloader:
            print('11111')
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().view(-1, 1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataloader.dataset)
        val_loss, precision, recall, f1, roc_auc = evaluate(model, criterion, val_dataloader, device)
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Training Loss: {epoch_loss:.4f}, '
              f'Validation Loss: {val_loss:.4f}, '
              f'Precision: {precision:.4f}, '
              f'Recall: {recall:.4f}, '
              f'F1-score: {f1:.4f}, '
              f'ROC AUC: {roc_auc:.4f}')

        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), 'best_model.pth')

# X_dnn = torch.tensor(X).float()
# y_dnn = torch.tensor(y).float()
X_pos = X[y == 1]
y_pos = y[y == 1]
X_neg = X[y == 0]
y_neg = y[y == 0]

X_pos = torch.tensor(X_pos).float()
y_pos = torch.tensor(y_pos).float()
X_neg = torch.tensor(X_neg).float()
y_neg = torch.tensor(y_neg).float()


# Set the minimum number of positive samples per split
min_pos_samples = 50
# Randomly split the positive samples
pos_train, pos_val = torch.utils.data.random_split(MyDataset(X_pos, y_pos),
                                   [len(X_pos) - min_pos_samples, min_pos_samples],
                                   generator=torch.Generator().manual_seed(42))

# Randomly split the negative samples
neg_train, neg_val = torch.utils.data.random_split(MyDataset(X_neg, y_neg),
                                   [len(X_neg) - len(pos_train), len(pos_train)],
                                   generator=torch.Generator().manual_seed(42))


# Concatenate the positive and negative splits to create your train and validation sets
train_dataset = torch.utils.data.ConcatDataset([pos_train, neg_train])
val_dataset = torch.utils.data.ConcatDataset([pos_val, neg_val])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

input_size = x.shape[1]

# X_dnn = torch.tensor(X).float()
# y_dnn = torch.tensor(y).float()
# # Split the data into training and validation sets
# dataset = TensorDataset(X_dnn, y_dnn)
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders for the training and validation sets

model = BinaryClassifier(input_size)
# commonly use in binary classification
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = 'cpu'

train(model, criterion,optimizer, train_loader, val_loader, device)
