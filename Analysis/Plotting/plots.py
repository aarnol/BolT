import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str)
parser.add_argument("-a", "--atlas", type=str)
parser.add_argument("-n", "--name", type=str)

argv = parser.parse_args()

import os 
import sys 

# sys.path.append(os.path.join(os.getcwd(), "/../Analysis/Logs"))

path = os.path.relpath(os.path.join("..", "Logs", "{}_{}_{}".format(argv.dataset, argv.atlas, argv.name)))

if os.path.exists(path):
	print(os.listdir(path))
else:
	print(f"Directory {path} does not exist.")
	
import pickle
import numpy as np
with open(os.path.join(path, "step_metrics.pkl"), "rb") as f:
	step_metrics = pickle.load(f)
with open(os.path.join(path, "epoch_metrics.pkl"), "rb") as f:
	epoch_metrics = pickle.load(f)
with open(os.path.join(path, "test_metrics.pkl"), "rb") as f:
	test_metrics = pickle.load(f)
with open(os.path.join(path, "test_results.pkl"), "rb") as f:
	test_results = pickle.load(f)

accuracy = [metric['accuracy'] for metric in step_metrics]
precision = [metric['precision'] for metric in step_metrics]
recall = [metric['recall'] for metric in step_metrics]
roc = [metric['roc'] for metric in step_metrics]
loss = [metric['loss'] for metric in step_metrics]

import matplotlib.pyplot as plt
# Plot accuracy over time
plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
plt.plot(accuracy, label='Accuracy')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.title('Accuracy over Time')
plt.legend()

# Plot precision over time
plt.subplot(2, 3, 2)
plt.plot(precision, label='Precision')
plt.xlabel('Step')
plt.ylabel('Precision')
plt.title('Precision over Time')
plt.legend()

# Plot recall over time
plt.subplot(2, 3, 3)
plt.plot(recall, label='Recall')
plt.xlabel('Step')
plt.ylabel('Recall')
plt.title('Recall over Time')
plt.legend()

# Plot ROC over time
plt.subplot(2, 3, 4)
plt.plot(roc, label='ROC')
plt.xlabel('Step')
plt.ylabel('ROC')
plt.title('ROC over Time')
plt.legend()

# Plot loss over time
plt.subplot(2, 3, 5)
plt.plot(loss, label='Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss over Time')
plt.legend()

# Save the plot to a file
plt.savefig(os.path.join(path, "metrics_over_time.png"))



# Function to average metrics over every 10 steps
def average_metrics(metrics, step=10):
    return [np.mean(metrics[i:i + step]) for i in range(0, len(metrics), step)]

# Averaging the metrics
avg_accuracy = average_metrics(accuracy)
avg_precision = average_metrics(precision)
avg_recall = average_metrics(recall)
avg_roc = average_metrics(roc)
avg_loss = average_metrics(loss)

# Plot averaged accuracy over time
plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
plt.plot(avg_accuracy, label='Avg Accuracy')
plt.xlabel('Step (x10)')
plt.ylabel('Avg Accuracy')
plt.title('Avg Accuracy over Time')
plt.legend()

# Plot averaged precision over time
plt.subplot(2, 3, 2)
plt.plot(avg_precision, label='Avg Precision')
plt.xlabel('Step (x10)')
plt.ylabel('Avg Precision')
plt.title('Avg Precision over Time')
plt.legend()

# Plot averaged recall over time
plt.subplot(2, 3, 3)
plt.plot(avg_recall, label='Avg Recall')
plt.xlabel('Step (x10)')
plt.ylabel('Avg Recall')
plt.title('Avg Recall over Time')
plt.legend()

# Plot averaged ROC over time
plt.subplot(2, 3, 4)
plt.plot(avg_roc, label='Avg ROC')
plt.xlabel('Step (x10)')
plt.ylabel('Avg ROC')
plt.title('Avg ROC over Time')
plt.legend()

# Plot averaged loss over time
plt.subplot(2, 3, 5)
plt.plot(avg_loss, label='Avg Loss')
plt.xlabel('Step (x10)')
plt.ylabel('Avg Loss')
plt.title('Avg Loss over Time')
plt.legend()

# Save the averaged plot to a file
plt.savefig(os.path.join(path, "avg_metrics_over_time.png"))

# Extract metrics from test_metrics
test_accuracy = [metric['accuracy'] for metric in test_metrics]
test_precision = [metric['precision'] for metric in test_metrics]
test_recall = [metric['recall'] for metric in test_metrics]
test_roc = [metric['roc'] for metric in test_metrics]
test_loss = [result['loss'] for result in test_results]

# Print the metrics
print("Test Accuracy:", test_accuracy)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)
print("Test ROC:", test_roc)
print("Test Loss:", test_loss)

plt.figure(figsize=(15, 10))

# Plot test accuracy over time
plt.subplot(2, 3, 1)
plt.plot(test_accuracy, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy over Time')
plt.legend()

# Plot test loss over time
plt.subplot(2, 3, 2)
plt.plot(test_loss, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Test Loss over Time')
plt.legend()

# Plot test precision over time
plt.subplot(2, 3, 3)
plt.plot(test_precision, label='Test Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Test Precision over Time')
plt.legend()

# Plot test recall over time
plt.subplot(2, 3, 4)
plt.plot(test_recall, label='Test Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Test Recall over Time')
plt.legend()

# Plot test ROC over time
plt.subplot(2, 3, 5)
plt.plot(test_roc, label='Test ROC')
plt.xlabel('Epoch')
plt.ylabel('ROC')
plt.title('Test ROC over Time')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(path, "loss_metrics.png"))

