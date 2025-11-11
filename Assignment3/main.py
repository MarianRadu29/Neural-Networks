import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

# https://d2l.ai/chapter_optimization/lr-scheduler.html#multi-factor-scheduler

learning_rate = 0.4
batch_size = 32
epochs = 50

np.random.seed(42)

best_acc = 0.0
patience = 5
patience_counter = 0

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1.0 + np.exp(-z)) 

def derivative_sigmoid(z: np.ndarray) -> np.ndarray:
    return sigmoid(z) * (1.0 - sigmoid(z))
 
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True) # avoiding overflow for large values
    exp_z = np.exp(z)
    return exp_z / (exp_z.sum(axis=1, keepdims=True) )

def forward(batch_data, w1,w2, b1,b2):
    """
    batch_data: (batch_size, 784)
    w1: (784, 100)
    w2: (100, 10)
    b1: (100,)
    b2: (10,)
    """
    z1 = batch_data @ w1 + b1  
    y1 = sigmoid(z1)     
    z2 = y1 @ w2 + b2    
    y_pred = softmax(z2) 
    return z2 , y_pred 

def one_hot_encode(labels, num_classes=10):
    """ 
    labels: (batch_size,) 
    """
    batch_size = labels.shape[0]
    one_hot = np.zeros((batch_size, num_classes), dtype=np.float64)
    # for all row in batch, set the column corresponding to the label to 1
    one_hot[np.arange(batch_size), labels] = 1.0
    return one_hot


def training(batch_data, batch_labels, w1,w2, b1,b2):
    """
    batch_data: (batch_size, 784)
    batch_labels: (batch_size,)
    w1: (784, 100)
    w2: (100, 10)
    b1: (100,)
    b2: (10,)
    """
    batch_size = batch_data.shape[0]

    # Forward pass
    _,y_pred = forward(batch_data, w1,w2, b1,b2)  # (batch_size, 10)

    # One-hot encode labels 
    y_true = one_hot_encode(batch_labels, num_classes=10)  # (batch_size, 10)


    #Backward pass
    z1 = batch_data @ w1 + b1  # (batch_size,100)

    # Gradient: Target - y
    # output layer
    delta2 = y_true - y_pred  # (batch_size, 10)
    dW2 = (sigmoid(z1).T @ delta2) / batch_size   
    db2 = np.sum(delta2, axis=0) / batch_size                

    # hidden layer
    delta1 = (delta2 @ w2.T) * derivative_sigmoid(z1)     # (batch_size,10) @  (10,100) * (batch_size,100)    
    dW1 = (batch_data.T @ delta1) / batch_size                        
    db1 = np.sum(delta1, axis=0) / batch_size               

    return dW1,dW2,db1,db2 


def predict(data, w1,w2,b1, b2):
    """
    data: (data_size, 784)
    w1: (784, 100)
    w2: (100, 10)
    b1: (100,)
    b2: (10,)
    """
    z1 = data @ w1 + b1
    y1 = sigmoid(z1)  
    z2 = y1 @ w2 + b2    
    return z2,np.argmax(z2, axis=1)  # (data_size,)

train_file = "Assignment3/data/extended_mnist_train.pkl"
test_file = "Assignment3/data/extended_mnist_test.pkl"


with open(train_file, "rb") as fp:
    train = pickle.load(fp)

with open(test_file, "rb") as fp:
    test = pickle.load(fp)


# Normalize data: pixel values from [0, 255] to [0, 1]
train_images = np.array([img.flatten() / 255.0 for img, _ in train], dtype=np.float64)
train_labels = np.array([label for _, label in train], dtype=np.int64)

train_images,validation_images = train_images[:int(9/10 * len(train_images))], train_images[int(9/10 * len(train_images)):]
train_labels,validation_labels = train_labels[:int(9/10 * len(train_labels))], train_labels[int(9/10 * len(train_labels)):]

test_images = np.array([img.flatten() / 255.0 for img, _ in test], dtype=np.float64)
test_labels = np.array([label for _, label in test], dtype=np.int64)


# Initialize weights and bias
b1 = np.zeros((100,), dtype=np.float64)
b2 = np.zeros((10,), dtype=np.float64)

# Xavier/Glorot initialization 
limit_w1 = np.sqrt(6.0 / (784 + 100))
limit_w2 = np.sqrt(6.0 / (100 + 10))

w1 = np.random.uniform(-limit_w1, limit_w1, (784, 100)).astype(np.float64)
w2 = np.random.uniform(-limit_w2, limit_w2, (100, 10)).astype(np.float64)


for epoch in range(epochs):
    length = len(train_images)
    
    # controlled shuffling for image and label arrays
    indices = np.arange(length)
    np.random.shuffle(indices)
    train_images_shuffled = train_images[indices]
    train_labels_shuffled = train_labels[indices]

    # Split data into batches efficiently 
    image_batches = np.array_split(train_images_shuffled, length // batch_size)
    label_batches = np.array_split(train_labels_shuffled, length // batch_size)

    for batch_data, batch_labels in zip(image_batches, label_batches):
        dW1,dW2,db1,db2 = training(batch_data, batch_labels, w1,w2, b1,b2)

        w1 += learning_rate * dW1
        w2 += learning_rate * dW2
        b1 += learning_rate * db1
        b2 += learning_rate * db2

    _,val_pred = predict(test_images, w1, w2,b1,b2)
    val_acc = (val_pred == test_labels).mean()
    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            learning_rate *= 0.75
            patience_counter = 0

    loss = F.cross_entropy(torch.tensor(forward(validation_images, w1,w2, b1,b2)[0]), torch.tensor(validation_labels)).item()
    print(f"epoch {epoch+1}: acc = {val_acc * 100:.2f}% best = {best_acc * 100:.2f}% lr = {learning_rate:.5f} loss = {loss:.5f}")


print("\n\n\n")
_ , predictions = predict(test_images, w1, w2,b1,b2)
print(f"Test acc = { (predictions == test_labels).mean() * 100:.2f}%")
print("\n\n\n")


df = pd.DataFrame({
    "ID": np.arange(len(predictions)),
    "target": predictions
})
df.to_csv("submission.csv", index=False)


