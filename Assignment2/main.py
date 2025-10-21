import pickle
import pandas as pd
import numpy as np


learning_rate = 0.1
num_batches = int(60_000 // 32)
epochs = 50


def softmax(z: np.ndarray) -> np.ndarray:
    """
    z: (batch_size, 10)
    """
    z = z - z.max(axis=1, keepdims=True) # avoiding overflow for large values
    exp_z = np.exp(z)
    return exp_z / (exp_z.sum(axis=1, keepdims=True) )


def training(batch_data, batch_labels, weights, bias):
    """
    batch_data: (batch_size, 784)
    batch_labels: (batch_size,)
    weights: (784, 10)
    bias: (10,)
    """
    batch_size = batch_data.shape[0]

    z = batch_data @ weights + bias  # (batch_size, 10)
    y_pred = softmax(z)  # (batch_size, 10)

    # One-hot encode labels 
    y_true = np.zeros((batch_size, 10), dtype=np.float64)
    # for all row in batch, set the column corresponding to the label to 1
    y_true[np.arange(batch_size), batch_labels] = 1.0

    # Gradient: Target - y
    d = y_true - y_pred  # (batch_size, 10)

    # Compute gradients 
    delta = (batch_data.T @ d) / batch_size  # (784,batch_size)*(batch_size, 10) = (784, 10)
    beta = np.sum(d, axis=0) / batch_size # (10,)

    return delta , beta 


def predict(data, W, b):
    """
    data: (data_size, 784)
    """
    z = data @ W + b  # (data_size, 10)
    return np.argmax(z, axis=1)  # (data_size,)

# if run from RN folder add `Assignment2` prefix
train_file = "Assignment2/data/extended_mnist_train.pkl"
test_file = "Assignment2/data/extended_mnist_test.pkl"


with open(train_file, "rb") as fp:
    train = pickle.load(fp)

with open(test_file, "rb") as fp:
    test = pickle.load(fp)


# Normalize data: pixel values from [0, 255] to [0, 1]
train_images = np.array([img.flatten() / 255.0 for img, _ in train], dtype=np.float64)
train_labels = np.array([label for _, label in train], dtype=np.int64)


length = len(train)
test_images = np.array([img.flatten() / 255.0 for img, _ in test], dtype=np.float64)
test_labels = np.array([label for _, label in test], dtype=np.int64)



# Initialize weights and bias
bias = np.zeros((10,), dtype=np.float64)
weights = (0.05 * np.random.randn(784, 10)).astype(np.float64)


for epoch in range(epochs):
    if epoch % 50 == 0:
        learning_rate *= 7/10
    # controlled shuffling for image and label arrays
    indices = np.arange(len(train_images))
    np.random.shuffle(indices)
    train_images_shuffled = train_images[indices]
    train_labels_shuffled = train_labels[indices]

    # Split data into batches efficiently 
    image_batches = np.array_split(train_images_shuffled, num_batches)
    label_batches = np.array_split(train_labels_shuffled, num_batches)

    for batch_data, batch_labels in zip(image_batches, label_batches):
        delta, beta = training(batch_data, batch_labels, weights, bias)
        # Update conform formulei din pdf
        weights += learning_rate * delta
        bias += learning_rate * beta

    val_pred = predict(test_images, weights, bias)
    val_acc = (val_pred == test_labels).mean()
    print(f"epoch {epoch+1}: acc = {val_acc * 100:.2f}%")


print("\n\n\n")
predictions = predict(test_images, weights, bias)
print(f"Test acc = { (predictions == test_labels).mean() * 100:.2f}%")
print("\n\n\n")


df = pd.DataFrame({
    "ID": np.arange(len(predictions)),
    "target": predictions
})
df.to_csv("submission.csv", index=False)


