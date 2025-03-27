import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from tabulate import tabulate

from utils.utils_gnn import array_to_graph
from utils.GCN import GCN

import matplotlib.pyplot as plt


# Params
basepath = './'
basepath_data = os.path.join(basepath, 'data/')
basepath_results = os.path.join(basepath, 'results')

CT_CONTROL = -1
CT_RRMS = 0
CT_SPMS = 1
CT_PPMS = 2


target_class = np.load('data/target_class.npy')
target = np.load('data/target_540.npy')
print(target_class.shape, target.shape)
target = target+1
print('target_class', np.unique(target_class, return_counts=True))
print('target', np.unique(target, return_counts=True))


## Data
data = np.load("./data/data540/FA_threshold.npy")
print("Data Shape      : {}".format(data.shape))  # Output: (270, 76, 76)
print("Min - Max values: {:.4f} - {:.4f}".format(np.min(data), np.max(data)))

prop = np.where(target == 1)[0].shape[0] / target.shape[0]
print("% of pwMS: {:.4f}".format(prop))

# Node embeddings
# Load the embeddings matrix
node_embeddings = np.load(os.path.join(basepath_data, 'data540/all_combined_data.npy'))
print(node_embeddings.shape)

constant_columns = []

for col in range(node_embeddings.shape[2]):
    if np.all(node_embeddings[:, :, col] == node_embeddings[0, 0, col]):
        constant_columns.append(col)

# Eliminar aquestes columnes de totes les matrius
if constant_columns:
    node_embeddings = np.delete(node_embeddings, constant_columns, axis=2)

print(f"Columnes constants eliminades: {constant_columns}")
print(f"Nova forma de node_embeddings: {node_embeddings.shape}")

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
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn as nn

def train(model, train_graphs, optimizer, loss_fn, device):
    model.train()
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    loss_all = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        label = batch.y  # Sense +1!
        loss = loss_fn(output, label)
        loss.backward()
        loss_all += batch.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_graphs)

# Funcions d'anàlisi de mètriques per classe (del primer fitxer)
def calculate_class_metrics(confusion_matrix, class_idx):
    true_positive = confusion_matrix[class_idx, class_idx]
    false_positive = np.sum(confusion_matrix[:, class_idx]) - true_positive
    false_negative = np.sum(confusion_matrix[class_idx, :]) - true_positive
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def convert_to_binary_cm(cm):
    binary_cm = np.zeros((2, 2))
    
    # True Positives for Healthy class (correctly classified as healthy)
    binary_cm[0, 0] = cm[0, 0]
    
    # False Negatives for Healthy class (healthy classified as ill)
    binary_cm[0, 1] = np.sum(cm[0, 1:])
    
    # False Positives for Healthy class (ill classified as healthy)
    binary_cm[1, 0] = np.sum(cm[1:, 0])
    
    # True Positives for Ill class (correctly classified as any ill type)
    binary_cm[1, 1] = np.sum(cm[1:, 1:])
    
    return binary_cm

def calculate_binary_metrics(binary_cm):
    # For Healthy class (0)
    tp_healthy = binary_cm[0, 0]
    fp_healthy = binary_cm[1, 0]
    fn_healthy = binary_cm[0, 1]
    
    precision_healthy = tp_healthy / (tp_healthy + fp_healthy) if (tp_healthy + fp_healthy) > 0 else 0
    recall_healthy = tp_healthy / (tp_healthy + fn_healthy) if (tp_healthy + fn_healthy) > 0 else 0
    f1_healthy = 2 * (precision_healthy * recall_healthy) / (precision_healthy + recall_healthy) if (precision_healthy + recall_healthy) > 0 else 0
    
    # For Ill class (1)
    tp_ill = binary_cm[1, 1]
    fp_ill = binary_cm[0, 1]
    fn_ill = binary_cm[1, 0]
    
    precision_ill = tp_ill / (tp_ill + fp_ill) if (tp_ill + fp_ill) > 0 else 0
    recall_ill = tp_ill / (tp_ill + fn_ill) if (tp_ill + fn_ill) > 0 else 0
    f1_ill = 2 * (precision_ill * recall_ill) / (precision_ill + recall_ill) if (precision_ill + recall_ill) > 0 else 0
    
    # Overall accuracy
    accuracy = (tp_healthy + tp_ill) / np.sum(binary_cm)
    
    return {
        "Healthy": {"Precision": precision_healthy, "Recall": recall_healthy, "F1": f1_healthy},
        "Ill": {"Precision": precision_ill, "Recall": recall_ill, "F1": f1_ill},
        "Accuracy": accuracy
    }

skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=8192)

NUM_EPOCHS = 1200
all_fold_preds = []
all_fold_test_indices = []
fold = 0

# Per emmagatzemar mètriques de cada fold
fold_cms = []
fold_auc_rocs = {}
classes = ["Control", "RRMS", "SPMS", "PPMS"]
multiclass_results = []
binary_results = []

for train_index, test_index in skf.split(data, target):
    test_index = test_index[test_index < 270]  # ens assegurem de que el test només treballi amb les dades reals

    # Si no hi ha dades de test després del filtre, continua amb la següent iteració
    if len(test_index) == 0:
        continue

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

    # Configuració del model amb sortida de 4 classes
    input_dim = node_embeddings.shape[2]
    num_classes = 4  # Assegurar coincidència amb les classes reals
    model = GCN(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    # Entrenament DINS del bucle fold
    loss_values = []
    for epoch in range(NUM_EPOCHS):
        loss_value = train(model, train_graphs, optimizer, loss_fn, device)
        loss_values.append(loss_value)
        if epoch % 50 == 0:  # printegem les epoques cada 50
            print(f"Epoch {epoch+1}: Loss {loss_value:.4f}")

    # Prova DINS del bucle fold
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    model.eval()
    fold_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            probs = F.softmax(output, dim=1).cpu().numpy()
            fold_preds.append(probs)
    
    # Collect all predictions for this fold
    fold_preds = np.vstack(fold_preds)
    all_fold_preds.append((test_index, fold_preds))
    all_fold_test_indices.extend(test_index)
    
    # Evaluate for this fold
    preds_class = np.argmax(fold_preds, axis=1)
    
    # Calcular matriu de confusió per aquest fold
    conf_matrix = confusion_matrix(y_test, preds_class)
    fold_cms.append(conf_matrix)
    print("Confusion Matrix for Fold {}:".format(fold))
    print(conf_matrix)
    
    # For multiclass, calculate per-class metrics
    if fold_preds.shape[1] > 2:  # Si és multiclasse
        try:
            auc_roc = roc_auc_score(y_test, fold_preds, multi_class='ovr')
            fold_auc_rocs[f"Fold {fold}"] = auc_roc
            print(f"Fold {fold} - AUC-ROC: {auc_roc:.4f}")
        except ValueError as e:
            print(f"Error calculating AUC-ROC for fold {fold}: {e}")
    
    # Anàlisi multiclasse detallat (del primer fitxer)
    for class_idx in range(4):
        precision, recall, f1 = calculate_class_metrics(conf_matrix, class_idx)
        multiclass_results.append([f"Fold {fold}", classes[class_idx], precision, recall, f1])
    
    # Anàlisi binari (Healthy vs. Ill)
    binary_cm = convert_to_binary_cm(conf_matrix)
    binary_metrics = calculate_binary_metrics(binary_cm)
    
    binary_results.append([
        f"Fold {fold}", 
        binary_metrics["Healthy"]["Precision"], 
        binary_metrics["Healthy"]["Recall"], 
        binary_metrics["Healthy"]["F1"],
        binary_metrics["Ill"]["Precision"], 
        binary_metrics["Ill"]["Recall"], 
        binary_metrics["Ill"]["F1"],
        binary_metrics["Accuracy"]
    ])
    
    # Classification Report per aquest fold
    print("\nClassification Report for Fold {}:".format(fold))
    print(classification_report(y_test, preds_class, zero_division=0))
    
    # Save fold results to file
    with open(filepath, "a") as file:
        file.write(f"Fold: {fold}\n")
        file.write("Train set size     : {}\n".format(X_train.shape))
        file.write("Test set size      : {}\n".format(X_test.shape))
        file.write("Train set % of pwMS: {:.4f} ({})\n".format(prop_train, y_train.sum()))
        file.write("Test set % of pwMS : {:.4f} ({})\n".format(prop_test, y_test.sum()))
        file.write("Confusion Matrix:\n{}\n".format(conf_matrix))
        file.write("Classification Report:\n{}\n\n".format(
            classification_report(y_test, preds_class, zero_division=0)
        ))

    # Plot training loss for this fold
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), loss_values, label=f'Training Loss - Fold {fold}', color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Function Evolution - Fold {fold}')
    plt.legend()
    plt.savefig(os.path.join(basepath_results, f"{current_time.strftime('%Y%m%d_%H%M%S')}_loss_fold_{fold}.png"))
    plt.close()

# Combine all fold predictions (for samples that were in test sets)
combined_preds = np.zeros((data.shape[0], num_classes))
combined_mask = np.zeros(data.shape[0], dtype=bool)

for test_indices, fold_predictions in all_fold_preds:
    for i, idx in enumerate(test_indices):
        combined_preds[idx] = fold_predictions[i]
        combined_mask[idx] = True

# Filter to only include samples that were in at least one test set
filtered_preds = combined_preds[combined_mask]
filtered_targets = target[combined_mask]

# Convert to final class predictions
final_class_preds = np.argmax(filtered_preds, axis=1)

# Compute final confusion matrix
final_conf_matrix = confusion_matrix(filtered_targets, final_class_preds)
print("\nFinal Confusion Matrix (all folds combined):")
print(final_conf_matrix)

# Final classification report
print("\nFinal Classification Report (all folds combined):")
print(classification_report(filtered_targets, final_class_preds, zero_division=0))

# AUC-ROC calculation for multiclass
try:
    # For multiclass, we need one-hot encoded targets for proper ROC calculation
    from sklearn.preprocessing import label_binarize
    
    # Get unique classes from the filtered targets
    unique_classes = np.unique(filtered_targets)
    
    # Binarize the labels for ROC calculation
    y_bin = label_binarize(filtered_targets, classes=unique_classes)
    
    # Calculate ROC AUC for each class
    auc_roc = roc_auc_score(y_bin, filtered_preds[:, :len(unique_classes)], multi_class='ovr')
    print(f"Final AUC-ROC (all folds combined): {auc_roc:.4f}")
    
    # Plot ROC curve for each class
    plt.figure(figsize=(10, 8))
    
    for i, cls in enumerate(unique_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], filtered_preds[:, i])
        plt.plot(fpr, tpr, lw=2, label=f'Class {cls} (AUC = {roc_auc_score(y_bin[:, i], filtered_preds[:, i]):.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Multiclass)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(basepath_results, f"{current_time.strftime('%Y%m%d_%H%M%S')}_ROC_multiclass.png"))
    plt.close()
    
except ValueError as e:
    print(f"Error calculating ROC curves: {e}")
    print("This might happen if not all classes are present in the test sets.")

# Save final results
with open(filepath, "a") as file:
    file.write("\n--- FINAL RESULTS (All Folds Combined) ---\n")
    file.write("Final Confusion Matrix:\n{}\n".format(final_conf_matrix))
    file.write("\nFinal Classification Report:\n{}\n".format(
        classification_report(filtered_targets, final_class_preds, zero_division=0)
    ))
    try:
        file.write(f"Final AUC-ROC: {auc_roc:.4f}\n")
    except:
        file.write("Could not calculate final AUC-ROC (possible class imbalance issue)\n")
    
    # Add accuracy
    accuracy = accuracy_score(filtered_targets, final_class_preds)
    file.write(f"Final Accuracy: {accuracy:.4f}\n")

# Afegim l'anàlisi detallat multiclasse com al primer fitxer
print("\n=== Detailed Multiclass Analysis ===\n")

# Display multiclass results
headers = ["Fold", "Class", "Precision", "Recall", "F1 Score"]
print(tabulate(multiclass_results, headers=headers, floatfmt=".4f"))

# Calculate average metrics across all folds for each class
avg_results = []
for class_idx in range(4):
    class_results = [result for result in multiclass_results if result[1] == classes[class_idx]]
    avg_precision = np.mean([result[2] for result in class_results])
    avg_recall = np.mean([result[3] for result in class_results])
    avg_f1 = np.mean([result[4] for result in class_results])
    avg_results.append(["Average", classes[class_idx], avg_precision, avg_recall, avg_f1])

print("\nAverage Metrics Across All Folds:")
print(tabulate(avg_results, headers=headers, floatfmt=".4f"))

# Calculate overall average metrics (macro average)
overall_avg_precision = np.mean([result[2] for result in avg_results])
overall_avg_recall = np.mean([result[3] for result in avg_results])
overall_avg_f1 = np.mean([result[4] for result in avg_results])

print("\nOverall Macro Average:")
print(f"Precision: {overall_avg_precision:.4f}")
print(f"Recall: {overall_avg_recall:.4f}")
print(f"F1 Score: {overall_avg_f1:.4f}")

# Calculate weighted average metrics based on class frequency
class_counts = [0, 0, 0, 0]
for fold_cm in fold_cms:
    for class_idx in range(4):
        class_counts[class_idx] += np.sum(fold_cm[class_idx, :])

total_samples = sum(class_counts)
weights = [count / total_samples for count in class_counts]

weighted_avg_precision = np.sum([avg_result[2] * weights[i] for i, avg_result in enumerate(avg_results)])
weighted_avg_recall = np.sum([avg_result[3] * weights[i] for i, avg_result in enumerate(avg_results)])
weighted_avg_f1 = np.sum([avg_result[4] * weights[i] for i, avg_result in enumerate(avg_results)])

print("\nWeighted Average (by class frequency):")
print(f"Precision: {weighted_avg_precision:.4f}")
print(f"Recall: {weighted_avg_recall:.4f}")
print(f"F1 Score: {weighted_avg_f1:.4f}")
print(f"Average AUC-ROC: {np.mean(list(fold_auc_rocs.values())):.4f}")

# Afegim l'anàlisi binari (Healthy vs. Ill)
print("\n\n=== Binary Classification (Control vs. Ill) ===\n")

# Display binary results
headers = ["Fold", "Precision (H)", "Recall (H)", "F1 (H)", "Precision (I)", "Recall (I)", "F1 (I)", "Accuracy"]
print(tabulate(binary_results, headers=headers, floatfmt=".4f"))

# Calculate average binary metrics
avg_precision_h = np.mean([result[1] for result in binary_results])
avg_recall_h = np.mean([result[2] for result in binary_results])
avg_f1_h = np.mean([result[3] for result in binary_results])
avg_precision_i = np.mean([result[4] for result in binary_results])
avg_recall_i = np.mean([result[5] for result in binary_results])
avg_f1_i = np.mean([result[6] for result in binary_results])
avg_accuracy = np.mean([result[7] for result in binary_results])

print("\nAverage Metrics Across All Folds:")
print(f"Control - Precision: {avg_precision_h:.4f}, Recall: {avg_recall_h:.4f}, F1: {avg_f1_h:.4f}")
print(f"Ill - Precision: {avg_precision_i:.4f}, Recall: {avg_recall_i:.4f}, F1: {avg_f1_i:.4f}")
print(f"Accuracy: {avg_accuracy:.4f}")

# Calculate weighted average metrics (weighted by class frequency)
total_healthy = 0
total_ill = 0

for fold_cm in fold_cms:
    total_healthy += np.sum(fold_cm[0, :])
    total_ill += np.sum(fold_cm[1:, :])

total_samples = total_healthy + total_ill
weight_healthy = total_healthy / total_samples
weight_ill = total_ill / total_samples

weighted_avg_precision = (avg_precision_h * weight_healthy) + (avg_precision_i * weight_ill)
weighted_avg_recall = (avg_recall_h * weight_healthy) + (avg_recall_i * weight_ill)
weighted_avg_f1 = (avg_f1_h * weight_healthy) + (avg_f1_i * weight_ill)

print("\nWeighted Average (by class frequency):")
print(f"Precision: {weighted_avg_precision:.4f}")
print(f"Recall: {weighted_avg_recall:.4f}")
print(f"F1 Score: {weighted_avg_f1:.4f}")

# Guardar els resultats de l'anàlisi detallat al fitxer
with open(filepath, "a") as file:
    file.write("\n\n=== Detailed Multiclass Analysis ===\n")
    
    # Multiclass results
    file.write("\nDetailed Metrics by Class and Fold:\n")
    file.write(tabulate(multiclass_results, headers=headers, floatfmt=".4f") + "\n")
    
    file.write("\nAverage Metrics Across All Folds:\n")
    file.write(tabulate(avg_results, headers=headers, floatfmt=".4f") + "\n")
    
    file.write("\nOverall Macro Average:\n")
    file.write(f"Precision: {overall_avg_precision:.4f}\n")
    file.write(f"Recall: {overall_avg_recall:.4f}\n")
    file.write(f"F1 Score: {overall_avg_f1:.4f}\n")
    
    file.write("\nWeighted Average (by class frequency):\n")
    file.write(f"Precision: {weighted_avg_precision:.4f}\n")
    file.write(f"Recall: {weighted_avg_recall:.4f}\n")
    file.write(f"F1 Score: {weighted_avg_f1:.4f}\n")
    file.write(f"Average AUC-ROC: {np.mean(list(fold_auc_rocs.values())):.4f}\n")
    
    # Binary results
    file.write("\n\n=== Binary Classification (Control vs. Ill) ===\n")
    
    file.write("\nBinary Classification Metrics by Fold:\n")
    file.write(tabulate(binary_results, headers=headers, floatfmt=".4f") + "\n")
    
    file.write("\nAverage Metrics Across All Folds:\n")
    file.write(f"Control - Precision: {avg_precision_h:.4f}, Recall: {avg_recall_h:.4f}, F1: {avg_f1_h:.4f}\n")
    file.write(f"Ill - Precision: {avg_precision_i:.4f}, Recall: {avg_recall_i:.4f}, F1: {avg_f1_i:.4f}\n")
    file.write(f"Accuracy: {avg_accuracy:.4f}\n")
    
    file.write("\nWeighted Average (by class frequency):\n")
    file.write(f"Precision: {weighted_avg_precision:.4f}\n")
    file.write(f"Recall: {weighted_avg_recall:.4f}\n")
    file.write(f"F1 Score: {weighted_avg_f1:.4f}\n")
