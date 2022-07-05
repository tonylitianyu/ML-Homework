import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

def transform_data(features):
    """
    Data can be transformed before being put into a linear discriminator. If the data
    is not linearly separable, it can be transformed to a space where the data
    is linearly separable, allowing the perceptron algorithm to work on it. This
    function should implement such a transformation on a specific dataset (NOT 
    in general).

    Args:
        features (np.ndarray): input features
    Returns:
        transformed_features (np.ndarray): features after being transformed by the function
    """


    new_features = []
    for i in range(0,len(features)):
        example = features[i]
        new_x1 = np.sqrt(np.sqrt(np.power(example[0],2.0) + np.power(example[1], 2.0)))
        new_x2 = 0.0#np.power(example[1], 2.0)

        new_features.append([new_x1, new_x2])

    new_features = np.array(new_features)

    
    return np.array(new_features)


class Perceptron():
    def __init__(self, max_iterations=200):
        """
        This implements a linear perceptron for classification. A single
        layer perceptron is an algorithm for supervised learning of a binary
        classifier. The idea is to draw a linear line in the space that separates
        the points in the space into two partitions. Points on one side of the 
        line are one class and points on the other side are the other class.


        Args:
            max_iterations (int): the perceptron learning algorithm stops after 
            this many iterations if it has not converged.

        """
        self.max_iterations = max_iterations
        self.w = np.array([0])


    def calculate_label(self, single_example):
        x = np.insert(single_example, 0, 1.0)
        res = np.dot(self.w, x)
        label = 1
        if res < 0:
            label = -1

        return label
        

    def all_prediction_match(self, features, targets):

        for i in range(0,len(features)):
            label = self.calculate_label(features[i])
            if targets[i] != label:
                return False

        return True

    def fit(self, features, targets):
        """
        Fit a single layer perceptron to features to classify the targets, which
        are classes (-1 or 1). This function should terminate either after
        convergence (dividing line does not change between interations) or after
        max_iterations (defaults to 200) iterations are done. Here is pseudocode for 
        the perceptron learning algorithm:

        begin initialize weights
            while not converged or not exceeded max_iterations
                for each example in features
                    if example is misclassified using weights
                    then weights = weights + example * label_for_example
            return weights
        end
        
        Note that label_for_example is either -1 or 1.

        Use only numpy to implement this algorithm. 

        Args:
            features (np.ndarray): 2D array containing inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (saves model and training data internally)
        """
        m = features.shape[0]
        self.w = np.random.rand(features.shape[1] + 1)

        count = 0
        k = -1
        while count < self.max_iterations and not self.all_prediction_match(features, targets):
            for k in range(0,len(features)):
                if self.calculate_label(features[k]) != targets[k]:
                    x = np.insert(features[k], 0, 1.0)
                    self.w = self.w + x * targets[k]

            count += 1



    def predict(self, features):
        """
        Given features, a 2D numpy array, use the trained model to predict target 
        classes. Call this after calling fit.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """

        predictions = []
        for i in range(0, len(features)):
            label = self.calculate_label(features[i])
            predictions.append(label)

        return predictions
        

    def visualize(self, features, targets):
        """
        This function should produce a single plot containing a scatter plot of the
        features and the targets, and the perceptron fit by the model should be
        graphed on top of the points.

        DO NOT USE plt.show() IN THIS FUNCTION.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (plots to the active figure)
        """
        plt.scatter(features[:,0], features[:,1])
        
        min_feature_1 = np.min(features[:,0])
        max_feature_1 = np.max(features[:,0])
        feature_1 = np.linspace(min_feature_1, max_feature_1, num=int((max_feature_1 - min_feature_1)/0.01))

        feature_2 = []

        for i in range(len(feature_1)):
            x2 = (0 - self.w[0] - self.w[1]*feature_1[i])/self.w[2]
            feature_2.append(x2)

        feature_2 = np.array(feature_2)
        

        plt.plot(feature_1, feature_2,'r-')

        plt.savefig('transform_me.png')
        

# import load_json_data
# if __name__ == '__main__':
#     features, targets = load_json_data.load_json_data('../data/transform_me.json')
#     features = transform_data(features)
#     p = Perceptron(max_iterations=100)
    

#     p.fit(features, targets)
#     targets_hat = p.predict(features)

#     print(np.allclose(targets, targets_hat))

#     p.visualize(features, targets)
