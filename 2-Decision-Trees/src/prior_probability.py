import numpy as np

class PriorProbability():
    def __init__(self):
        """
        This is a simple classifier that only uses prior probability to classify 
        points. It just looks at the classes for each data point and always predicts
        the most common class.

        """
        self.most_common_class = None

    def fit(self, features, targets):
        """
        Implement a classifier that works by prior probability. Takes in features
        and targets and fits the features to the targets using prior probability.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N 
                examples.
        Output:
            VOID: You should be updating self.most_common_class with the most common class
            found from the prior probability.
        """
        # #evidence
        # p_evi = []
        # features = np.array(features)
        # for i in range(len(features[0])):
        #     p_evi_per_feature = (features[:,i] == 1).sum()/len(features)
        #     p_evi.append(p_evi_per_feature)

        # p_evi = np.array(p_evi)
        # evidence = np.prod(p_evi)


        #likelihood
        unique_class, class_count = np.unique(targets, return_counts=True)
        
        self.most_common_class = unique_class[np.argmax(class_count)]




    def predict(self, data):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        Outputs:
            predictions (np.array): numpy array of size N array which has the predicitons 
            for the input data.
        """
        predicitons = []
        for i in range(data.shape[0]):
            predicitons.append(self.most_common_class)

        return np.array(predicitons)
