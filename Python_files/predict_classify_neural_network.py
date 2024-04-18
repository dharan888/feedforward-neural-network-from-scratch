import numpy as np

from neural_network import Network
from layers import fcLayer, classLayer, regLayer

# training data
x_train = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
y_train = np.array([[0, 1, 1, 0]])

# network
fcnet = Network()
fcnet.add(fcLayer(2, 3, 'tanh'))
fcnet.add(regLayer(3, 1, 'tanh'))

# train
[train_loss, test_loss] = fcnet.fit(x_train, y_train, x_train, y_train, batch_size=1, epochs=1000, learning_rate=0.1)

# test
predictions = fcnet.predict(x_train)
print("predictions:")
print(predictions)
