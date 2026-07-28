import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import pandas as pd

font_path = 'simhei.ttf'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NewsDataset(Dataset):
    def __init__(self, file_path, label_map, max_seq_length=256):
        super().__init__()
        self.examples = []
        self.max_seq_length = max_seq_length
        data = pd.read_csv(file_path, sep=',', header=0)  
        for _, row in data.iterrows():
            label, text = row[1], row[2]
            if label not in label_map:
                continue
            label_id = label_map[label]
            token_ids = self.tokenize_text(text)
            self.examples.append((token_ids, label_id))

    def tokenize_text(self, text):
        tokens = [ord(char) % 1000 for char in text]
        if len(tokens) < self.max_seq_length:
            tokens += [0] * (self.max_seq_length - len(tokens))
        else:
            tokens = tokens[:self.max_seq_length]
        return torch.tensor(tokens)

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)

def custom_collate_fn(batch):
    input_ids, labels = zip(*batch)
    input_ids = torch.stack(input_ids)
    labels = torch.tensor(labels)
    return input_ids.to(device), labels.to(device)

class SimpleCNNClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes):
        super(SimpleCNNClassifier, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.conv1 = torch.nn.Conv2d(1, 100, (3, embed_size))
        self.conv2 = torch.nn.Conv2d(1, 100, (4, embed_size))
        self.conv3 = torch.nn.Conv2d(1, 100, (5, embed_size))
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(300, num_classes)

    def conv_and_pool(self, x, conv):
        x = torch.relu(conv(x)).squeeze(3)
        x = torch.nn.functional.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x1 = self.conv_and_pool(x, self.conv1)
        x2 = self.conv_and_pool(x, self.conv2)
        x3 = self.conv_and_pool(x, self.conv3)
        x = torch.cat((x1, x2, x3), 1)
        x = self.dropout(x)
        out = self.fc(x)
        return out

label_map = {
    '事实': 0, '谣言': 1
}

max_seq_length = 256
full_dataset = NewsDataset('data.csv', label_map, max_seq_length=max_seq_length)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

batch_size = 256
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)

vocab_size = 1000
embed_size = 128
num_classes = len(label_map)
model = SimpleCNNClassifier(vocab_size, embed_size, num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, labels in data_loader:
            logits = model(input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, axis=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    report = classification_report(all_labels, all_preds, target_names=list(label_map.keys()))
    cm = confusion_matrix(all_labels, all_preds)
    
    return total_loss / len(data_loader), accuracy, recall, precision, f1, report, cm

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for input_ids, labels in train_loader:
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, axis=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_accuracy = total_correct / total_samples
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    test_loss, test_accuracy, _, _, _, _, _ = evaluate(model, test_loader)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Accuracy={train_accuracy:.4f}, Test Loss={test_loss:.4f}, Test Accuracy={test_accuracy:.4f}")

# 最终测试评估
test_loss, test_accuracy, test_recall, test_precision, test_f1, test_report, test_cm = evaluate(model, test_loader)

print("Test Classification Report:")
print(test_report)
print("Test Confusion Matrix:")
print(test_cm)
print(f"Test Accuracy={test_accuracy:.4f}, Recall={test_recall:.4f}, Precision={test_precision:.4f}, F1 Score={test_f1:.4f}")

def plot_and_save_metrics(train_metrics, test_metrics, title, ylabel, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(train_metrics, label='Train')
    plt.plot(test_metrics, label='Test')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(filename)
    plt.close()

plot_and_save_metrics(train_losses, test_losses, 'Training and Test Loss', 'Loss', './cnn_model/loss_plot.png')
plot_and_save_metrics(train_accuracies, test_accuracies, 'Training and Test Accuracy', 'Accuracy', './cnn_model/accuracy_plot.png')

font_prop = FontProperties(fname=font_path, size=14)
def plot_and_save_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix', fontproperties=font_prop)
    plt.ylabel('True Label', fontproperties=font_prop)
    plt.xlabel('Predicted Label', fontproperties=font_prop)

    plt.xticks(fontproperties=font_prop, size=12)
    plt.yticks(fontproperties=font_prop, size=12)

    plt.savefig('./cnn_model/confusion_matrix.png')
    plt.close()

class_names = list(label_map.keys())
plot_and_save_confusion_matrix(test_cm, class_names)

output_dir = './cnn-trained_model'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
torch.save(model.state_dict(), os.path.join(output_dir, 'cnn_model.pth'))
print("save:", output_dir)
