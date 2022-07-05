import numpy as np

class Node():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Node classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

class DecisionTree():
    def __init__(self, attribute_names):
        """
        TODO: Implement this class.

        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Node classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)
        
        """
        self.attribute_names = attribute_names
        self.tree = None

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def find_max_gain(self, attributes, features, targets):
        max_gain_attr_idx = 0
        for a in range(0, len(attributes)):
            curr_max = information_gain(features, max_gain_attr_idx, targets)
            curr_att = information_gain(features, a, targets)

            if curr_max < curr_att:
                max_gain_attr_idx = a

        best_att_name = attributes[max_gain_attr_idx]

        
        return best_att_name, max_gain_attr_idx   #attribute name, attribute index with max info gain


    def splitData(self, data, data_in_attribute, attribute_val):
        D_a_idx = np.argwhere(np.abs(data_in_attribute - attribute_val) < 0.001)
        D_a_idx = np.squeeze(D_a_idx)
        D_a = data[D_a_idx, :]
        if len(D_a.shape) == 1:
            D_a = np.array([D_a])

        
        return D_a

    def ID3(self, data, attributes):


        node = Node()

        #find the most common class in targets
        targets = data[:,-1]
        class_count = np.zeros(2)
        for ta in targets:
            class_count[int(ta)] += 1
        max_class_Idx = np.argmax(class_count)
        node.value = max_class_Idx

        
        
        #base case

        #If targets have only one class, return this node
        non_zero_idx = np.nonzero(class_count)
        if len(non_zero_idx[0]) == 1:
            node.attribute_name = 'leaf'
            return node

        #If no attributes, return this node
        if len(attributes) == 0:
            node.attribute_name = 'leaf'
            return node


        #Get features from data
        features = data[:,0:-1]
        #Get targets from data
        targets = data[:,-1].reshape((1,-1))

        #Find the best split attribute
        a_name, a_idx = self.find_max_gain(attributes, features, targets)

        #Assign best attribute name to this node name
        node.attribute_name = a_name

        #Get all the data column for this attribute
        data_in_attribute = features[:, a_idx]

        #left branch: no (0)
        #right branch: yes (1)

        for att in range(0,2):
            D_a_origin = self.splitData(data, data_in_attribute, att)
            D_a = np.delete(D_a_origin, a_idx, 1)
            temp_att_0 = attributes.copy()
            temp_att_0.remove(a_name)

            childNode = Node()
            if len(D_a) == 0:
                childNode = Node(value=max_class_Idx, attribute_name='leaf')
            else:
                childNode = self.ID3(D_a, temp_att_0)

            node.branches.append(childNode)



        return node
            




    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        Output:
            VOID: It should update self.tree with a built decision tree.
        """
        self._check_input(features)

        data = np.hstack((features, targets))
        self.tree = self.ID3(data, self.attribute_names)


    def predict(self, features):
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
        self._check_input(features)

        predicitons = np.zeros(features.shape[0])
        for p in range(len(predicitons)):
            sample = features[p,:]
            t = self.tree
            
            while len(t.branches) != 0:
                
                curr_att_idx = self.attribute_names.index(t.attribute_name)
                value = int(sample[curr_att_idx])
                t = t.branches[value]

            predicitons[p] = t.value

        return predicitons

                

    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)


def get_entropy_from_feature(feature_col):
    unique_class, class_count = np.unique(feature_col, return_counts=True)
    total_count = feature_col.shape[0]
    HS_entropy_sum = 0
    for c in range(len(unique_class)):
        prior = class_count[c]/total_count
        prior_entropy = prior*np.log2(prior)
        HS_entropy_sum += prior_entropy

    HS_entropy_sum = -HS_entropy_sum

    return HS_entropy_sum


def information_gain(features, attribute_index, targets):
    """
    TODO: Implement me!

    Information gain is how a decision tree makes decisions on how to create
    split points in the tree. Information gain is measured in terms of entropy.
    The goal of a decision tree is to decrease entropy at each split point as much as
    possible. This function should work perfectly or your decision tree will not work
    properly.

    Information gain is a central concept in many machine learning algorithms. In
    decision trees, it captures how effective splitting the tree on a specific attribute
    will be for the goal of classifying the training data correctly. Consider
    data points S and an attribute A. S is split into two data points given binary A:

        S(A == 0) and S(A == 1)

    Together, the two subsets make up S. If A was an attribute perfectly correlated with
    the class of each data point in S, then all points in a given subset will have the
    same class. Clearly, in this case, we want something that captures that A is a good
    attribute to use in the decision tree. This something is information gain. Formally:

        IG(S,A) = H(S) - H(S|A)

    where H is information entropy. Recall that entropy captures how orderly or chaotic
    a system is. A system that is very chaotic will evenly distribute probabilities to
    all outcomes (e.g. 50% chance of class 0, 50% chance of class 1). Machine learning
    algorithms work to decrease entropy, as that is the only way to make predictions
    that are accurate on testing data. Formally, H is defined as:

        H(S) = sum_{c in (classes in S)} -p(c) * log_2 p(c)

    To elaborate: for each class in S, you compute its prior probability p(c):

        (# of elements of class c in S) / (total # of elements in S)

    Then you compute the term for this class:

        -p(c) * log_2 p(c)

    Then compute the sum across all classes. The final number is the entropy. To gain
    more intution about entropy, consider the following - what does H(S) = 0 tell you
    about S?

    Information gain is an extension of entropy. The equation for information gain
    involves comparing the entropy of the set and the entropy of the set when conditioned
    on selecting for a single attribute (e.g. S(A == 0)).

    For more details: https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics

    Args:
        features (np.array): numpy array containing features for each example.
        attribute_index (int): which column of features to take when computing the
            information gain
        targets (np.array): numpy array containing labels corresponding to each example.

    Output:
        information_gain (float): information gain if the features were split on the
            attribute_index.
    """
    
    targets = np.transpose(targets)

    HS_entropy_sum = get_entropy_from_feature(targets)


    curr_feature = features[:, attribute_index]
    unique_att, att_count = np.unique(curr_feature, return_counts=True)

    att_prob = []
    for a in range(len(unique_att)):
        att_prob.append(att_count[a]/features.shape[0])
    att_prob = np.array(att_prob)

    posi_idx = []
    nega_idx = []
    for r in range(features.shape[0]):
        if features[r][attribute_index] == 1:
            posi_idx.append(r)
        else:
            nega_idx.append(r)

    posi_branch = targets[posi_idx,:]

    nega_branch = targets[nega_idx,:]

    branch_entropy = 0.0
    if len(att_prob) == 1:
        branch_entropy = att_prob[0] * get_entropy_from_feature(nega_branch) + att_prob[0] * get_entropy_from_feature(posi_branch)

    else:
        posi_entropy = att_prob[1] * get_entropy_from_feature(posi_branch)
        nega_entropy = att_prob[0] * get_entropy_from_feature(nega_branch)

        branch_entropy = posi_entropy + nega_entropy

    IG = HS_entropy_sum - branch_entropy


    return IG



# from data import load_data, train_test_split

if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['GoodGrades', 'GoodLetters', 'GoodSAT', 'IsRich', 'HasScholarship', 'ParentAlum', 'SchoolActivities']

    # features, targets, attribute_names = load_data('../data/PlayTennis.csv')
    # train_features, train_targets, test_features, test_targets = train_test_split(features, targets, 1.0)
    decision_tree = DecisionTree(attribute_names=attribute_names)
    #decision_tree.fit(train_features, train_targets)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Node(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Node(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
