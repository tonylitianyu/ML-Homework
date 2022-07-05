## Problems

## Code implementation (5 points)
Pass test cases by implementing the functions in the `src` directory.

Your grade for this section is defined by the autograder. If it says you got an 80/100, you get 4 points here.

## Free response questions (5 points)

1. Assume you have a k-Nearest Neighbor (kNN) classifier for doing a 2-way classification. Assume it uses an <img src="/tex/09af92d48ab87fa468ebde78082d1091.svg?invert_in_darkmode&sanitize=true&sanitize=true" align=middle width=17.96371994999999pt height=22.465723500000017pt/> norm distance metric (e.g., Euclidean, Manhattan, etc.).
Assume a straightforward implementation, like the one discussed in class. What is the time complexity to select a class label for (i.e., to classify) a new point using this model?Give your answer in terms of the number of points, <img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/>, in the data set. You can answer this using big O notation and explain why you think that is the answer, or you can answer this by simply explaining your reasoning in terms of the number of points. What is the space complexity of the model, in terms of <img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/> and the number of dimensions, <img src="/tex/2103f85b8b1477f430fc407cad462224.svg?invert_in_darkmode&sanitize=true&sanitize=true" align=middle width=8.55596444999999pt height=22.831056599999986pt/>, in the vector representing each data point? Once again, you can use big O notation with an explanation or give an explanation of your reasoning. 

2. What is the time complexity of training a kNN classifier in terms of the number of points in the training data, <img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/>, and the number of dimensions, <img src="/tex/2103f85b8b1477f430fc407cad462224.svg?invert_in_darkmode&sanitize=true&sanitize=true" align=middle width=8.55596444999999pt height=22.831056599999986pt/>, in the vector representing each data point? The same note about big O notation applies here as did for Question 1.

3. Is a kNN classifier able to learn a non-linear decision surface? Why or why not? 

4. Imagine we're building a *collaborative filter*, which we'll learn a lot about later in the quarter. Collaborative filters are essentially how recommendation algorithms work on sites like Amazon ("people who bought blank also bought blank") and Netflix ("you watched blank, so you might also like blank"). They work by comparing distances between users. If two users are similar, then items that one user has seen and liked but the other hasn't seen are recommended to the other user. First, frame this problem as a kNN regression problem. How are users compared to each other? Given the k nearest neighbors of a user, how can these k neighbors be used to estimate the rating for a movie that the user has not seen?

5. Your boss gives you a very large dataset (i.e. many, many training samples). What problems do you see coming up with implementing kNN? Name a few of them. 

6. Discuss possible solutions to the problems you have identified in Question 5.
 
7. You are given these six training examples as ((a_i, b_i), y_i) values, where a_i and b_i are the two feature values (positive integers) and y_i is the class label: {((1,1),-1),((2, 2), −1), ((2, 8), +1), ((4, 4), +1), ((6, 5), −1), ((3, 6), −1)}. Classify the following test example at coordinates (4, 7) using a kNN classifier with k = 3 and Manhattan distance defined by d((u, v), (p, q)) = |u −p| + |v −q|. Explain shortly how you came up with the answer.
 
8. Assume we increase k in a kNN classifier from 1 to n, where n is the number of training examples. Discuss if the classification accuracy on the training set increases. Consider weighted and unweighted KNN classifiers

9. Can you think of a strategy how to define the K-parameter in the KNN-algorithm? If you cannot find a strategy, please explain why you think this is a hard problem.

10. What happens if the data-dimensionality (i.e. the number of features) becomes really large? What problems might occur? For high-dimensonal data, what is the importance of having a lot training data? NOTE: We will revisit this problem in a later lecture, but it's for you to already think about this problem.

NOTE: The peer mentors will assign a grade according to the overall quality of your report. A rough estimate might be that each question is worth about 0.25 points.
