import numpy as np

class Regularization:
    """
    Abstract base class for regularization terms in gradient descent.

    *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***

    Arguments:
        reg_param - (float) The hyperparameter that controls the amount of
            regularization to perform. Must be non-negative.
    """

    def __init__(self, reg_param=0.05):
        self.reg_param = reg_param

    def forward(self, w):
        """
        Implements the forward pass through the regularization term.

        *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***

        Arguments:
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            regularization_term - (float) The value of the regularization term
                evaluated at w.
        """
        pass

    def backward(self, w):
        """
        Implements the backward pass through the regularization term.

        *** THIS IS A BASE CLASS: YOU DO NOT NEED TO IMPLEMENT THIS ***

        Arguments:
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            gradient_term - (np.array) A numpy array of length d+1. The
                gradient of the regularization term evaluated at w.
        """
        pass


class L1Regularization(Regularization):
    """
    L1 Regularization for gradient descent.
    """

    def forward(self, w):
        """
        Implements the forward pass through the regularization term. For L1,
        this is the L1-norm of the model parameters weighted by the
        regularization parameter. Note that the bias (the last value in w)
        should NOT be included in regularization.

        Arguments:
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            regularization_term - (float) The value of the regularization term
                evaluated at w.
        """
        reg_w = w[:-1]
        return np.sum(np.abs(reg_w)) * self.reg_param

    def backward(self, w):
        """
        Implements the backward pass through the regularization term. The
        backward pass is the gradient of the forward pass with respect to the
        model parameters.

        Arguments:
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            gradient_term - (np.array) A numpy array of length d+1. The
                gradient of the regularization term evaluated at w.
        """
        back_w = np.zeros(len(w))
        for i in range(0,len(w)-1):
            if w[i] > 1:
                back_w[i] = 1
            else:
                back_w[i] = -1
        
        return back_w * self.reg_param


class L2Regularization(Regularization):
    """
    L2 Regularization for gradient descent.
    """

    def forward(self, w):
        """
        Implements the forward pass through the regularization term. For L2,
        this is half the squared L2-norm of the model parameters weighted by
        the regularization parameter. Note that the bias (the last value in w)
        should NOT be included in regularization.

        Arguments:
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            regularization_term - (float) The value of the regularization term
                evaluated at w.
        """

        reg_w = w[:-1]

        return 0.5 * np.sum(np.square(reg_w)) * self.reg_param

    def backward(self, w):
        """
        Implements the backward pass through the regularization term. The
        backward pass is the gradient of the forward pass with respect to the
        model parameters.

        Arguments:
            w - (np.array) A 1D array of parameters of length d+1. The current
                parameters learned by the model. The +1 refers to the bias
                term.
        Returns:
            gradient_term - (np.array) A numpy array of length d+1. The
                gradient of the regularization term evaluated at w.
        """
        w[-1] = 0.0
        return w * self.reg_param
