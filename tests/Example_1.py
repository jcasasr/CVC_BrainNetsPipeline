import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F

from utils.utils_gnn import array_to_graph
from utils.GCN import GCN


# Params
basepath = './'
basepath_data = os.path.join(basepath, 'data/')
basepath_results = os.path.join(basepath, 'results')

CT_CONTROL = -1
CT_RRMS = 0
CT_SPMS = 1
CT_PPMS = 2


target_class = np.load('data/target_class.npy')
target = np.load('data/target.npy')
print(target_class.shape, target.shape)
print(np.unique(target_class, return_counts=True))
print(np.unique(target, return_counts=True))

## Data
data = np.load("./data/FA_threshold.npy")
print("Data Shape      : {}".format(data.shape))  # Output: (270, 76, 76)
print("Min - Max values: {:.4f} - {:.4f}".format(np.min(data), np.max(data)))

prop = np.where(target == 1)[0].shape[0] / target.shape[0]
print("% of pwMS: {:.4f}".format(prop))

# Node embeddings
# Load the embeddings matrix
node_embeddings = np.load(os.path.join(basepath_data, 'node_embeddings.npy'))
print(node_embeddings.shape)

## Train
device = torch.device('cpu')

import datetime

# Get the current date and time
current_time = datetime.datetime.now()

# Format the current time as a string (e.g., "2024-12-11_14-30-00")
filename = current_time.strftime("%Y%m%d_%H%M%S") + ".txt"
filepath = os.path.join(basepath, "results/" + filename)

# Create and write to the file
with open(filepath, "w") as file:
    file.write(f'Model trained on {current_time.strftime("%Y %m %d %H:%M:%S")}.\n')

num_folds = 5

# Save parameters of model
with open(filepath, "a") as file:
    file.write("Data normal.\n")
    file.write(f"stratifiedKfold; {num_folds} folds.\n")
    file.write("Node embeddings dimension: 49.\n")
    file.write("1 conv layer (49, 128) + 3 lin layer (128, 64), (64, 16), (16, 2).\n\n")

from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
import torch.nn as nn

skf = StratifiedKFold(n_splits=num_folds, shuffle=True)

NUM_EPOCHS = 100
preds = np.zeros(data.shape[0])
fold = 0

for train_index, test_index in skf.split(data, target):
    fold += 1
    print("Fold: {}".format(fold))

    # split dataset
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = target[train_index], target[test_index]
    embed_train, embed_test = node_embeddings[train_index], node_embeddings[test_index]
    
    prop_train = np.where(y_train == 1)[0].shape[0] / y_train.shape[0]
    prop_test = np.where(y_test == 1)[0].shape[0] / y_test.shape[0]
    print("Train set size     : {}".format(X_train.shape))
    print("Test set size      : {}".format(X_test.shape))
    print("Train set % of pwMS: {:.4f} ({})".format(prop_train, y_train.sum()))
    print("Test set % of pwMS : {:.4f} ({})".format(prop_test, y_test.sum()))

    # list of Data structures (one for each subject)
    train_graphs = []
    for i in range(X_train.shape[0]):
        g = array_to_graph(X_train[i], embed_train[i], y_train[i])
        train_graphs.append(g)
        
    test_graphs = []
    for i in range(X_test.shape[0]):
        g = array_to_graph(X_test[i], embed_test[i], y_test[i])
        test_graphs.append(g)

    # create the model
    input_dim = node_embeddings.shape[2]
    model = GCN(input_dim)
    model = model.to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # Custom loss function to penalise false negatives
    #class_weights = torch.tensor([1.0, 1.0]).to(device) # Higher weight for minority class 0 (negative class)
    #loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # train function
    def train():
        model.train()
        
        train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)

        loss_all = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            label = batch.y
            label = F.one_hot(label, num_classes=2)
            label = label.type(torch.FloatTensor)
            label = label.to(device)
            loss = loss_fn(output, label)
            loss.backward()
            loss_all += batch.num_graphs * loss.item()
            optimizer.step()

        return loss_all / len(train_graphs)

    # train for N epochs
    for epoch in range(NUM_EPOCHS):
        loss_value = train()
        print("Train loss at epoch {}: {:.4f}".format(epoch + 1, loss_value))

    # test phase 
    test_loader = DataLoader(test_graphs, batch_size=len(test_graphs), shuffle=False)
    
    for batch in test_loader:
        batch = batch.to(device)
        test_preds = F.softmax(model(batch), dim=1).detach().numpy()
    

    # Collect predictions and true labels
    all_preds = []
    all_labels = []

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            # Get the predicted class (highest probability)
            preds_eval = torch.argmax(F.softmax(output, dim=1), dim=1).cpu().numpy()
            labels = batch.y.cpu().numpy()  # True labels
            all_preds.extend(preds_eval)
            all_labels.extend(labels)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Display confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)


    test_preds = test_preds[:, 1]
    preds[test_index] = test_preds
    
    auc_roc = roc_auc_score(y_test, test_preds)
    print("Test AUC: {:.2f}".format(auc_roc))

    # Open the text file in append mode (this will add to the file without overwriting)
    with open(filepath, "a") as file:
        file.write(f"Fold: {fold}\n")
        file.write("Train set size     : {}\n".format(X_train.shape))
        file.write("Test set size      : {}\n".format(X_test.shape))
        file.write("Train set % of pwMS: {:.4f} ({})\n".format(prop_train, y_train.sum()))
        file.write("Test set % of pwMS : {:.4f} ({})\n".format(prop_test, y_test.sum()))
        file.write("Confusion Matrix:\n{}\n".format(conf_matrix))
        file.write("Test AUC: {:.2f}\n\n".format(auc_roc))


auc_roc = roc_auc_score(target, preds)
auc_pr = average_precision_score(target, preds)
    
best_acc = 0
best_th = 0
for th in preds:
    acc = accuracy_score(target, (preds >= th).astype(int))
    if acc >= best_acc:
        best_acc = acc
        best_th = th
        
print("")
prop = np.where(target == 1)[0].shape[0] / target.shape[0]
print("% of pwMS: {:.4f}".format(prop))
print("AUC ROC  : {:.4f}".format(auc_roc))
print("AUC PR   : {:.4f}".format(auc_pr))
print("ACC      : {:.4f}".format(best_acc))

# Open the text file in append mode (this will add to the file without overwriting)
with open(filepath, "a") as file:
    file.write("% of pwMS: {:.4f}\n".format(prop))
    file.write("AUC ROC  : {:.4f}\n".format(auc_roc))
    file.write("AUC PR   : {:.4f}\n".format(auc_pr))
    file.write("ACC      : {:.4f}\n\n".format(best_acc))


## RESULTS
# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

# Test phase
test_loader = DataLoader(test_graphs, batch_size=len(test_graphs), shuffle=False)

# Collect predictions and true labels
all_preds = []
all_labels = []

model.eval()  # Set model to evaluation mode
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        output = model(batch)
        # Get the predicted class (highest probability)
        preds_eval = torch.argmax(F.softmax(output, dim=1), dim=1).cpu().numpy()
        labels = batch.y.cpu().numpy()  # True labels
        all_preds.extend(preds_eval)
        all_labels.extend(labels)

# Compute confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

# Display confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Optional: Classification Report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

# AUC-ROC
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams['font.size'] = 12


# compute AUC-ROC and ROC curve
auc_roc = roc_auc_score(target, preds)
fpr, tpr, ths = roc_curve(target, preds)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color="darkorange", lw=lw)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (AUC-ROC: {:.4f})".format(auc_roc))
plt.savefig("./results/" + current_time.strftime("%Y%m%d_%H%M%S") + "_ROC_" + ".png")
plt.show()

# Compute AUC-PR
auc_pr = average_precision_score(target, preds)
prec, recall, ths = precision_recall_curve(target, preds)

plt.figure()
lw = 2
plt.plot(recall, prec, color="darkorange", lw=lw)
plt.plot([0, 1], [prop, prop], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision Recall Curve (AUC-PR: {:.4f})".format(auc_pr))
plt.savefig("./results/" + current_time.strftime("%Y%m%d_%H%M%S") + "_PrRec_" + ".png")
plt.show()

# Compute ACC and threshold
best_acc = 0
best_th = 0
for th in ths:
    acc = accuracy_score(target, (preds >= th).astype(int))
    if acc >= best_acc:
        best_acc = acc
        best_th = th

plt.figure()
plt.scatter(target, preds, alpha=0.5, color="darkorange", lw=lw)
plt.plot([0, 1], [best_th, best_th], color='navy', lw=lw, linestyle='--')
plt.xlabel("Real value")
plt.ylabel("Predicted value")
plt.title("Scatter plot (Accuracy at best threshold: {:.4f})".format(best_acc))
plt.savefig("./results/" + current_time.strftime("%Y%m%d_%H%M%S") + "_scatt_" + ".png")
plt.show()
