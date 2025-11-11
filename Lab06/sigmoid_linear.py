import numpy as np
import matplotlib.pyplot as plt

# definim functia sigmoida
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# cream un set simplu de date pentru 2D
np.random.seed(0)
X = np.random.randn(100, 2)
w = np.array([1.5, -1])
b = 0.2

# calculam outputul neuronului si clasificarea (0/1)
z = X @ w + b
y = sigmoid(z)
y_class = (y > 0.5).astype(int)

# desenam punctele si separatorul liniar
x1_range = np.linspace(-3, 3, 100)
x2_boundary = -(w[0]*x1_range + b)/w[1]

plt.figure(figsize=(6,5))
plt.scatter(X[y_class==0,0], X[y_class==0,1], color='blue', label='Clasa 0')
plt.scatter(X[y_class==1,0], X[y_class==1,1], color='red', label='Clasa 1')
plt.plot(x1_range, x2_boundary, 'k--', label='Separator liniar')
plt.title('Neuron sigmoidal: separator liniar')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)
plt.show()
