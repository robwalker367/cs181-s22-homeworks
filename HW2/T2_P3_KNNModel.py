import numpy as np

# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.K = k

    def predict(self, X_pred):
        data = np.hstack([self.X, self.y.reshape((27, 1))])
        data = list(map(lambda x : list(x), data))
        yhats = []
        for xp in X_pred:
            k_closest = sorted(data, key=lambda x2, x1=xp : (self.__distance(x1, x2), x2))[0:self.K]
            yhat = np.argmax(np.bincount(list(map(lambda y : y[2], k_closest))))
            yhats.append(yhat)
        return np.array(yhats)

    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, X, y):
        self.X = X
        self.y = y

    def __distance(self, star1, star2):
        return ((star1[0] - star2[0]) / 3) ** 2 + (star1[1] - star2[1]) ** 2
