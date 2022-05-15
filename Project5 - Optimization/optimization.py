import math
import numpy as np


class Regressor:

    def __init__(self) -> None:
        self.X, self.y = self.generate_dataset(n_samples=200, n_features=1)
        self.n, d = self.X.shape
        self.w = np.zeros((d, 1))

    def generate_dataset(self, n_samples, n_features):
        """
        Generates a regression dataset
        Returns:
            X: a numpy.ndarray of shape (100, 2) containing the dataset
            y: a numpy.ndarray of shape (100, 1) containing the labels
        """
        from sklearn.datasets import make_regression

        np.random.seed(42)
        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=30)
        y = y.reshape(n_samples, 1)
        return X, y

    def linear_regression(self,X):
        """
        Performs linear regression on a dataset
        Returns:
            y: a numpy.ndarray of shape (n, 1) containing the predictions
        """
        y = np.dot(X, self.w)
        return y

    def predict(self, X):
        """
        Predicts the labels for a given dataset
        X: a numpy.ndarray of shape (n, d) containing the dataset
        Returns:
            y: a numpy.ndarray of shape (n,) containing the predictions
        """
        y = np.dot(X, self.w).reshape(X.shape[0])
        return y

    def compute_loss(self, X):
        """
        Computes the MSE loss of a prediction
        Returns:
            loss: the loss of the prediction
        """
        predictions = self.linear_regression(X)
        loss = np.mean((predictions - self.y) ** 2)
        return loss

    def compute_gradient(self , X , y):
        """
        Computes the gradient of the MSE loss
        Returns:
            grad: the gradient of the loss with respect to w
        """
        predictions = self.linear_regression(X)
        dif = (predictions - y)
        grad = 2 * np.dot(X.T, dif)
        return grad

    def fit(self, optimizer='adam_optimizer', n_iters=340 , render_animation=True):
        """
        Trains the model
        optimizer: the optimization algorithm to use
        X: a numpy.ndarray of shape (n, d) containing the dataset
        y: a numpy.ndarray of shape (n, 1) containing the labels
        n_iters: the number of iterations to train for
        """
        last_loss = math.inf
        figs = []
        g = 0
        dw = 0
        m = 0
        v = 0
        for i in range(1, n_iters + 1):

            if optimizer == 'gradient_descent':
                self.w = self.gradient_descent(alpha=0.003)
            elif optimizer == "sgd_optimizer":
                self.w = self.sgd_optimizer(alpha=0.01)
            elif optimizer == "sgd_momentum":
                self.w = self.sgd_momentum(alpha=0.0009, momentum=0.9)
            elif optimizer == "adagrad_optimizer":
                self.w, g, dw = self.adagrad_optimizer(135 , 0.01 , dw, g)
            elif optimizer == "rmsprop_optimizer":
                self.w, g, dw = self.rmsprop_optimizer(40, 0.9, 0.01 , dw, g)
            elif optimizer == "adam_optimizer":
                self.w, v, m, dw = self.adam_optimizer(0.1, 0.9, 0.01, 0.01 , dw, i , m, v)

            loss = self.compute_loss(self.X)
            if loss >= last_loss:
                print('best loss: ', loss)
                break
            else:
                last_loss = loss

            if i % 10 == 0:
                print("Iteration: ", i)
                print("Loss: ", loss)

            if render_animation:
                import matplotlib.pyplot as plt
                from moviepy.video.io.bindings import mplfig_to_npimage

                fig = plt.figure()
                plt.scatter(self.X, self.y, color='red')
                plt.plot(self.X, self.predict(self.X), color='blue')
                plt.xlim(self.X.min(), self.X.max())
                plt.ylim(self.y.min(), self.y.max())
                plt.title(f'Optimizer:{optimizer}\nIteration: {i}')
                plt.close()
                figs.append(mplfig_to_npimage(fig))

        if render_animation and len(figs) > 0:
            from moviepy.editor import ImageSequenceClip
            clip = ImageSequenceClip(figs, fps=5)
            clip.write_gif(f'{optimizer}_animation.gif', fps=5)

    def gradient_descent(self, alpha):
        """
        Performs gradient descent to optimize the weights
        alpha: the learning rate
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """
        grad = self.compute_gradient(self.X, self.y)
        dif = - alpha * grad
        self.w += dif
        return self.w

    def sgd_optimizer(self, alpha):
        """
        Performs stochastic gradient descent to optimize the weights
        alpha: the learning rate
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """
        batch_size = 20
        xy = np.c_[self.X.reshape(self.n, -1), self.y.reshape(self.n, 1)]
        r = np.random.default_rng(seed=42)
        r.shuffle(xy)
        w = self.w

        for first in range(0, self.n, batch_size):
            last = first + batch_size
            x_batch, y_batch = xy[first:last, :-1], xy[first:last, -1:]
            grad = self.compute_gradient(x_batch, y_batch)
            dif = - alpha * grad
            w += dif

        return w

    def sgd_momentum(self, alpha, momentum):
        """
        Performs SGD with momentum to optimize the weights
        alpha: the learning rate
        momentum: the momentum
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """
        batch_size = 20
        xy = np.c_[self.X.reshape(self.n, -1), self.y.reshape(self.n, 1)]
        r = np.random.default_rng(seed=42)
        r.shuffle(xy)
        w = self.w
        dif = 0
        for first in range(0, self.n, batch_size):
            last = first + batch_size
            x_batch, y_batch = xy[first:last, :-1], xy[first:last, -1:]
            grad = self.compute_gradient(x_batch, y_batch)
            dif = - alpha * grad + momentum * dif
            w += dif

        return w

    def adagrad_optimizer(self, alpha, epsilon , dw , g):
        """
        Performs Adagrad optimization to optimize the weights
        alpha: the learning rate
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        w = self.w
        grad = self.compute_gradient(self.X, self.y)
        dw += grad
        g += dw ** 2
        w += -(alpha/np.sqrt(g + epsilon)) * grad

        return self.w , g, dw


    def rmsprop_optimizer(self, alpha, beta, epsilon , dw, g):
        """
        Performs RMSProp optimization to optimize the weights
        g: sum of squared gradients
        alpha: the learning rate
        beta: the momentum
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        w = self.w
        grad = self.compute_gradient(self.X, self.y)
        dw += grad
        g = beta * g + (1- beta) * dw ** 2
        w += -(alpha/np.sqrt(g + epsilon)) * grad

        return self.w , g, dw

    def adam_optimizer(self, alpha, beta1, beta2, epsilon , dw , i , m, v):
        """
        Performs Adam optimization to optimize the weights
        m: the first moment vector
        v: the second moment vector
        alpha: the learning rate
        beta1: the first momentum
        beta2: the second momentum
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        w = self.w

        grad = self.compute_gradient(self.X, self.y)
        dw += grad

        m = beta1 * m + (1 - beta1) * dw
        v = beta2 * v + (1 - beta2) * dw ** 2

        m = m / (1 - beta1 ** (i + 1))
        v = v / (1 - beta2 ** (i + 1))

        w += -(alpha / np.sqrt(v + epsilon)) * m
        return self.w, v, m, dw

Regressor().fit()
