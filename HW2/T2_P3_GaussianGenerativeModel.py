import numpy as np
from scipy.stats import multivariate_normal as mvn  # you may find this useful


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance
        self.K = 3
        self.means = None
        self.covars = None
        self.covar = None
        self.priors = None

    # Implement this method!
    def fit(self, X, y):
        # Calculate class frequencies and optimal means
        freqs = np.zeros(self.K)
        means = np.zeros((self.K, X.shape[1]))
        for i in range(len(y)):
            klass = y[i]
            for j in range(self.K):
                if klass == j:
                    means[klass] += X[i]
            freqs[klass] += 1
        for klass in range(self.K):
            means[klass] /= freqs[klass]
        self.means = means

        # Calculate covariance matrix
        covars = [np.zeros((X.shape[1], X.shape[1])) for _ in range(self.K)]
        covar = np.zeros((X.shape[1], X.shape[1]))
        for i in range(len(X)):
            klass = y[i]
            for k in range(self.K):
                if klass == k:
                    c = np.dot(X[i].reshape(2,1) - means[j].reshape(2,1), (X[i].reshape(2,1) - means[k].reshape(2,1)).T)
                    covars[klass] += c
                    covar += c
                
        for k in range(self.K):
            covars[k] = covars[k] / freqs[k]
        self.covars = covars
        self.covar = covar / np.sum(freqs)

        # Calculate priors
        self.priors = freqs / np.sum(freqs)

    def predict(self, X_pred):
        yhats = np.zeros(X_pred.shape[0])
        for i in range(len(X_pred)):
            max_k, max_yhat = 0, 0
            for k in range(self.K):
                c = (self.covar if self.is_shared_covariance else self.covars[k])
                yhat = mvn.pdf(X_pred[i], mean=self.means[k].T, cov=c) * self.priors[k]
                if yhat > max_yhat:
                    max_k = k
                    max_yhat = yhat
            yhats[i] = max_k
        return yhats

    def negative_log_likelihood(self, X, y):
        ll = 0
        for i in range(len(y)):
            klass = y[i]
            for k in range(self.K):
                if klass == k:
                    c = (self.covar if self.is_shared_covariance else self.covars[k])
                    ll += np.log(mvn.pdf(X[i], mean=self.means[k].T, cov=c) * self.priors[k])
        return -1 * ll
