# Do not use anything outside of the standard distribution of python
# when implementing this class
import math
from time import sleep

from sklearn.feature_selection import SelectFdr 

class LogisticRegressionSGD:
    """
    Logistic regression with stochastic gradient descent
    """

    def __init__(self, eta, mu, n_feature):
        """
        Initialization of model parameters
        """
        self.eta = eta
        self.weight = [0.0] * n_feature
        self.mu = mu

    def fit(self, X, y):
        """
        Update model using a pair of training sample
        """
        gamma = math.fsum((self.weight[feature_id] * value for feature_id, value in X))

        sigmoid = 0.0
        if gamma >= 0:

            sigmoid = 1.0 / (1.0 + math.exp((-1.0)*gamma))
        else:
            sigmoid = 1.0 - 1.0 / (1.0 + math.exp((-1.0)*gamma))
        

        #   update the coefficient vector

        for feature_id, value in X:

            # 𝑤𝑡=𝑤𝑡−1+ 𝜂𝑥𝑡(𝑦𝑡 − 𝜎(𝑤𝑇𝑡−1𝑥𝑡)) 
 
 
            self.weight[feature_id] = self.weight[feature_id] + self.eta*( y - value* sigmoid )

        # apply penalty
        # 𝑤𝑡 =𝑤𝑡−1+𝜂( 𝑥𝑡(𝑦𝑡−𝜎(𝑤𝑇𝑥𝑡)) − 2𝜇𝑤)
        # 𝑤𝑡 =𝑤𝑡−1+ 𝜂(− 2𝜇𝑤)

        for feature_id, value in enumerate(self.weight):
            self.weight[feature_id] = self.weight[feature_id]- 2 *self.eta * self.mu * self.weight[feature_id]
 

    def predict(self, X):
        """
        Predict 0 or 1 given X and the current weights in the model
        """
        return 1 if self.predict_prob(X) > 0.5 else 0

    def predict_prob(self, X):
        """
        Sigmoid function
        """

        return 1.0 / (1.0 + math.exp(-math.fsum((self.weight[f]*v for f, v in X))))
