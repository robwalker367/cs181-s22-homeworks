import numpy as np
import matplotlib.pyplot as plt



# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam
        self.runs = 200000
        self.loss = np.zeros(self.runs)

    def fit(self, X, y):
        # Process X, y
        X = np.hstack([np.ones((len(X), 1)), X])
        y = self.__oneHotEncode(y)

        # Guess weights
        self.W = np.random.rand(3, 3)

        # Gradient descent
        N = len(X)
        for i in range(0, self.runs):
            yhat = self.__predict(X)
            gradient = X.T@(yhat - y) / N + 2 * self.lam * self.W
            self.loss[i] = -1 * np.trace(np.dot(np.log(yhat), y.T))
            self.W = self.W - self.eta * gradient

    def predict(self, X_pred):
        # Bias trick
        X_pred = np.hstack([np.ones((len(X_pred), 1)), X_pred])

        # Make prediction
        return np.argmax(self.__predict(X_pred), axis=1)


    def __predict(self, X):
        return self.__softmax(X@self.W)
    
    def __softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]

    def __oneHotEncode(self, v):
        o = np.zeros((v.size, v.max()+1))
        o[np.arange(v.size), v] = 1
        return o

    # TODO: Implement this method!
    def visualize_loss(self, output_file, show_charts=False):
        if show_charts:
            title = output_file + f'_eta{self.eta}_lam{self.lam}'
            plt.figure()
            plt.title(title)
            plt.xlabel('Runs')
            plt.ylabel('Cross-entropy loss')
            x = np.arange(1, self.runs+1)
            plt.plot(x, self.loss)
            plt.savefig(title + '.png')
            plt.show()
