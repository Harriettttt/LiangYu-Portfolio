import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import xgboost as xgb

font_path = 'simhei.ttf'

class SpamDataset(Dataset):
    def __init__(self, pass_file, spam_file, label_map, max_seq_length=256):
        super().__init__()
        self.examples = []
        self.max_seq_length = max_seq_length

        # data reading
        with open(pass_file, 'r', encoding='utf-8') as f:
            for line in f:
                label_id = label_map['pass']
                token_ids = self.tokenize_text(line.strip())
                self.examples.append((token_ids, label_id))

        with open(spam_file, 'r', encoding='utf-8') as f:
            for line in f:
                label_id = label_map['spam']
                token_ids = self.tokenize_text(line.strip())
                self.examples.append((token_ids, label_id))

    def tokenize_text(self, text):
        tokens = [ord(char) % 1000 for char in text]  # ASCII
        if len(tokens) < self.max_seq_length:
            tokens += [0] * (self.max_seq_length - len(tokens))
        else:
            tokens = tokens[:self.max_seq_length]
        return torch.tensor(tokens, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)

# label map
label_map = {
    'pass': 0, 'spam': 1
}

# data loading
max_seq_length = 256
full_dataset = SpamDataset('msgpass.log.seg', 'msgspam.log.seg', label_map, max_seq_length=max_seq_length)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# change into numpy
def dataset_to_numpy(dataset):
    data_loader = DataLoader(dataset, batch_size=len(dataset))
    for data in data_loader:
        inputs, labels = data
        return inputs.numpy(), labels.numpy()

train_inputs, train_labels = dataset_to_numpy(train_dataset)
test_inputs, test_labels = dataset_to_numpy(test_dataset)

# initialize
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(label_map), use_label_encoder=False, eval_metric='mlogloss')

# training xgboost
xgb_model.fit(train_inputs, train_labels)
print("XGBoost done")

# evaluate model
def evaluate(model, inputs, labels):
    preds = model.predict(inputs)
    accuracy = accuracy_score(labels, preds)
    recall = recall_score(labels, preds, average='macro')
    precision = precision_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    report = classification_report(labels, preds, target_names=list(label_map.keys()))
    cm = confusion_matrix(labels, preds)
    return accuracy, recall, precision, f1, report, cm

train_accuracy, train_recall, train_precision, train_f1, train_report, train_cm = evaluate(xgb_model, train_inputs, train_labels)
test_accuracy, test_recall, test_precision, test_f1, test_report, test_cm = evaluate(xgb_model, test_inputs, test_labels)

print("Train Classification Report:")
print(train_report)
print("Test Classification Report:")
print(test_report)

print(f"Test Accuracy={test_accuracy:.4f}, Recall={test_recall:.4f}, Precision={test_precision:.4f}, F1 Score={test_f1:.4f}")

# confusion matrix
font_prop = FontProperties(fname=font_path, size=14)
def plot_and_save_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix', fontproperties=font_prop)
    plt.ylabel('True Label', fontproperties=font_prop)
    plt.xlabel('Predicted Label', fontproperties=font_prop)

    plt.xticks(fontproperties=font_prop, size=12)
    plt.yticks(fontproperties=font_prop, size=12)

    if not os.path.exists('./xgb_model'):
        os.makedirs('./xgb_model')
    plt.savefig('./xgb_model/confusion_matrix.png')
    plt.close()

class_names = list(label_map.keys())
plot_and_save_confusion_matrix(test_cm, class_names)

# plot
def plot_and_save_accuracy(train_accuracy, test_accuracy, ylabel, filename):
    plt.figure(figsize=(10, 5))
    plt.plot([1, 2], [train_accuracy, test_accuracy], marker='o', label='Train/Test Accuracy')
    plt.xticks([1, 2], ['Train', 'Test'])
    plt.title(ylabel)
    plt.xlabel('Dataset')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(filename)
    plt.close()

plot_and_save_accuracy(train_accuracy, test_accuracy, 'Accuracy', './xgb_model/accuracy_plot.png')
