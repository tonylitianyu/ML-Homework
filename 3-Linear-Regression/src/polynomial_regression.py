import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

class PolynomialRegression():
    def __init__(self, degree):
        """
        Implement polynomial regression from scratch.
        
        This class takes as input "degree", which is the degree of the polynomial 
        used to fit the data. For example, degree = 2 would fit a polynomial of the 
        form:

            ax^2 + bx + c
        
        Your code will be tested by comparing it with implementations inside sklearn.
        DO NOT USE THESE IMPLEMENTATIONS DIRECTLY IN YOUR CODE. You may find the 
        following documentation useful:

        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

        Here are helpful slides:

        http://interactiveaudiolab.github.io/teaching/eecs349stuff/eecs349_linear_regression.pdf
    
        The internal representation of this class is up to you. Read each function
        documentation carefully to make sure the input and output matches so you can
        pass the test cases. However, do not use the functions numpy.polyfit or numpy.polval. 
        You should implement the closed form solution of least squares as detailed in slide 10
        of the lecture slides linked above.

        Usage:
            import numpy as np
            
            x = np.random.random(100)
            y = np.random.random(100)
            learner = PolynomialRegression(degree = 1)
            learner.fit(x, y) # this should be pretty much a flat line
            predicted = learner.predict(x)

            new_data = np.random.random(100) + 10
            predicted = learner.predict(new_data)

            # confidence compares the given data with the training data
            confidence = learner.confidence(new_data)


        Args:
            degree (int): Degree of polynomial used to fit the data.
        """


        self.degree = degree
        self.w = []

        
    
    def fit(self, features, targets):
        """
        Fit the given data using a polynomial. The degree is given by self.degree,
        which is set in the __init__ function of this class. The goal of this
        function is fit features, a 1D numpy array, to targets, another 1D
        numpy array.
        

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing real-valued targets.
        Returns:
            None (saves model and training data internally)
        """
        
        X = []
        for i in range(0, len(features)):
            x_row = [1.0]
            for d in range(1,self.degree+1):
                term = features[i]**d
                x_row.append(term)
            X.append(x_row)

        X = np.array(X)


        Y = np.array([targets]).T
        

        self.w = np.linalg.inv(X.T @ X) @ X.T @ Y
        self.w = self.w.flatten()



        

        
        


    def predict(self, features):
        """
        Given features, a 1D numpy array, use the trained model to predict target 
        estimates. Call this after calling fit.

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """
        predictions = []
        for i in range(len(features)):
            y = 0.0
            for d in range(self.degree+1):
                y += self.w[d] * (features[i]**d)

            predictions.append(y)

        return np.array(predictions)
            


    def visualize(self, features, targets):
        """
        This function should produce a single plot containing a scatter plot of the
        features and the targets, and the polynomial fit by the model should be
        graphed on top of the points.

        DO NOT USE plt.show() IN THIS FUNCTION. Instead, use plt.savefig().

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing real-valued targets.
        Returns:
            None (plots to the active figure)
        """
        plt.scatter(features, targets)
        
        min_features = np.min(features)
        max_features = np.max(features)
        all_features = np.linspace(min_features, max_features, num=int((max_features - min_features)/0.01))
        predictions = self.predict(all_features)
        plt.plot(all_features, predictions,'r-')
        plt.savefig('polynomial_fit.png')

# import generate_regression_data
# import metrics
# if __name__ == '__main__':
#     degree = 2
#     p = PolynomialRegression(degree)
#     x, y = generate_regression_data.generate_regression_data(degree, 100, amount_of_noise=0.0)

#     p.fit(x, y)
#     y_hat = p.predict(x)
#     mse = metrics.mean_squared_error(y,y_hat)
#     print(mse)
#     p.visualize(x,y)
