#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSC311H5F - Assignment 2
@author: Josh Alexander
UTORid: alexa364
Student Number: 1005434458
"""

import pickle
import bonnerlib2D as bl2d
import numpy as np
import matplotlib.pyplot as plt
import sklearn.utils as utl
import scipy.stats as sci

import sklearn.linear_model as lin
import sklearn.discriminant_analysis as da
import sklearn.naive_bayes as naiv
import sklearn.neural_network as nn

# RETRIEVE DATA
with open('cluster_data.pickle','rb') as file:
    dataTrain, dataTest = pickle.load(file)

Xtrain, Ttrain = dataTrain
Xtest, Ttest = dataTest

# QUESTION 1----------------------------------------
print("\nQuestion 1")
print("----------")

# Q1(a) ~~~~~~~~~
print("\nQ1(a):")
clf = lin.LogisticRegression(multi_class='multinomial', solver='lbfgs')
clf.fit(Xtrain, Ttrain) # train the model

clf_train_score = clf.score(Xtrain, Ttrain)
clf_test_score = clf.score(Xtest, Ttest)

print("Training Data Accuracy: ",clf_train_score)
print("Test Data Accuracy: ", clf_test_score)

# Q1(b) ~~~~~~~~~
def plot_dboundaries(clf, title):
    bl2d.plot_data(Xtrain, Ttrain)
    bl2d.boundaries(clf)
    plt.title(title)
    plt.show()


plot_dboundaries(clf, "Question 1(b): decision boundaries for linear classification")

# Q1(e) ~~~~~~~~~
print("\nQ1(e):")

def predict(X,W,b):
    """
    Multiclass classification predictions
    """
    n = b.shape[0] # number of classes
    z = W @ X.T + b.reshape(n,1) # z = Wx+b
    return np.argmax(z.T, axis=1) # take the highest score index


Y1 = clf.predict(Xtest)
Y2 = predict(Xtest, clf.coef_, clf.intercept_)
print("Squared Magnitude of Y1-Y2: ", np.sum((Y1-Y2) ** 2))

# Q1(f) ~~~~~~~~~
print("\nQ1(f):")

def one_hot(Tint):
    """
    Converts integer target values to one-hot encodings
    In dim: Tint -> [N]
    Out dim: Thot -> [N, J], where J is the number of classes
    """
    N = Tint.size
    Tint = Tint.reshape((N,1))
    C = np.arange(0, np.max(Tint)+1)
    return np.array((Tint == C), dtype=int)


print(one_hot(np.array([4,3,2,3,2,1,2,1,0])))


# QUESTION 2----------------------------------------
print("\nQuestion 2")
print("----------")

# Q2(a) ~~~~~~~~~
def softmax(z):
    z_exp = np.exp(z.T)
    denom = np.sum(z_exp, axis=1).reshape(z_exp.shape[0],1)
    return np.divide(z_exp, denom)


def softmax_regression(W, X):
    """
    Return the probability distribution (y) and prediction (pred)
    of the data (X) with weight matrix (W) using softmax regression
    """
    z = W @ X.T
    y = softmax(z)
    pred = np.argmax(z.T, axis=1)
    return y, pred


def calc_lce(y,thot):
    """
    Return the average of cross entropy loss
    """
    lce = np.log(y) @ -thot.T
    dg = np.identity(lce.shape[0])
    result = lce[dg == 1]
    return np.mean(result)


def calc_acc(pred,t):
    """
    Return the accuracy of the predictions
    """
    c = np.array((pred == t), dtype=int)
    return np.sum(c)/c.size


def prepare_data(X, T):
    """
    Extend data matrix and convert target values to one-hot vectors
    """
    extended_matrix = np.insert(X, 0, 1, axis=1)
    return extended_matrix, one_hot(T)


def initialize_weight(X, T):
    """
    Randomly initialize weight matrix (including the bias term)
    """
    K = np.max(T) # number of classes + 1
    D = X.shape[1] # number of features
    return np.random.randn(K + 1, D + 1) / 10000


def print_accuracy_diff(gd, train_accuracy, test_accuracy):
    print("\n{} Training Accuracy: {}".format(gd, train_accuracy[-1]))
    print("Q1(a) Training Accuracy: ", clf_train_score)
    print("Diff: ", train_accuracy[-1] - clf_train_score)
    print("\n{} Test Accuracy: {}".format(gd, test_accuracy[-1]))
    print("Q1(a) Test Accuracy: ", clf_test_score)
    print("Diff: ", test_accuracy[-1] - clf_test_score)


def show_plot(title, vlabel, hlabel):
    plt.xlabel(hlabel)
    plt.ylabel(vlabel)
    plt.title(title)
    plt.show()


def plot_figures(q, W, train_avg_ce, test_avg_ce, train_accuracy, test_accuracy):
    # vii
    plt.semilogx(train_avg_ce, color='b')
    plt.semilogx(test_avg_ce, color='r')
    show_plot(q + ": Training and test loss v.s. iterations", 
              "Cross entropy", 
              "Iteration number")
    
    # viii
    plt.semilogx(train_accuracy, color='b')
    plt.semilogx(test_accuracy, color='r')
    show_plot(q + ": Training and test accuracy v.s. iterations", 
              "Accuracy", 
              "Iteration number")
    
    # ix
    plt.semilogx(test_avg_ce[50:], color='r')
    show_plot(q + ": test loss from iteration 50 on",
              "Cross entropy", 
              "iteration number")
    
    # x
    plt.semilogx(train_avg_ce[50:], color='b')
    show_plot(q + ": training loss from iteration 50 on",
              "Cross entropy", 
              "iteration number")
    
    # xiii
    Wmatrix = W[:,1:]
    bias = W[:,0]
    bl2d.plot_data(Xtrain, Ttrain)
    bl2d.boundaries2(Wmatrix, bias, predict)
    plt.title(q+": decision boundaries for linear classification")
    plt.show()


def GDlinear(I, lrate):
    """
    Gradient descent for multi-class classification
    with cross-entropy loss function.
    
    I: number of iterations
    lrate : learning rate
    """
    np.random.seed(7)
    print("learning rate = ",lrate)
    
    train_x, train_t = prepare_data(Xtrain, Ttrain)
    test_x, test_t = prepare_data(Xtest, Ttest)
    
    N = Xtrain.shape[0] # number of entries
    W = initialize_weight(Xtrain, Ttrain)
    
    train_avg_ce, train_accuracy = [], []
    test_avg_ce, test_accuracy = [], []
    
    for c in range(I):
        
        train_y, train_pred = softmax_regression(W, train_x)
        test_y, test_pred = softmax_regression(W, test_x)
        
        if c > 0:
            train_avg_ce.append(calc_lce(train_y, train_t))
            test_avg_ce.append(calc_lce(test_y, test_t))
            train_accuracy.append(calc_acc(train_pred, Ttrain))
            test_accuracy.append(calc_acc(test_pred, Ttest))
        
        W = W - (lrate / N) * ((train_y - train_t).T @ train_x) # weights update
        
    print_accuracy_diff("GDlinear", train_accuracy, test_accuracy)
    plot_figures("Question 2(a)", W, train_avg_ce, test_avg_ce, train_accuracy, test_accuracy)
    

print("\nQ2(a):")
#GDlinear(10000, 0.1)

# Q2(d) ~~~~~~~~~
def SGDlinear(I, batch_size, lrate0, alpha, kappa):
    """
    Stochastic gradient descent for multi-class classification
    with cross-entropy loss function
    
    I: number of epochs
    batch_size: number of training points in each mini batch
    lrate0: initial learning rate
    alpha: decay rate
    kappa: burn-in period
    """
    np.random.seed(7)
    print("Batch size: ", batch_size)
    print("Initial learning rate: ", lrate0)
    print("Decay rate: ", alpha)
    print("Burn-in period: ", kappa)
    
    train_x, train_t = prepare_data(Xtrain, Ttrain)
    test_x, test_t = prepare_data(Xtest, Ttest)
    
    N = Xtrain.shape[0] # number of entries
    W = initialize_weight(Xtrain, Ttrain)
    
    train_avg_ce, train_accuracy = [], []
    test_avg_ce, test_accuracy = [], []
    
    lrate = lrate0
    nsweep = int(np.ceil(N/batch_size))
    
    for i in range(I):      
        # [epoch begin]
        
        strain_x, strain_t = utl.shuffle(train_x, train_t)
        stest_x, stest_t = utl.shuffle(test_x, test_t)
        
        if i > kappa:
            k = alpha * (i - kappa)
            lrate = lrate0 / (1 + k)
        
        for s in range(nsweep):
            # creating current mini batch
            start = s * batch_size
            end = start + batch_size
            batch_x = train_x[start:end,:]
            batch_t = train_t[start:end,:]
            
            batch_y, batch_pred = softmax_regression(W, batch_x)
            W = W - (lrate / N) * ((batch_y - batch_t).T @ batch_x) # weights update
    
        train_y, train_pred = softmax_regression(W, train_x)
        test_y, test_pred = softmax_regression(W, test_x)
    
        train_avg_ce.append(calc_lce(train_y, train_t))
        test_avg_ce.append(calc_lce(test_y, test_t))
        train_accuracy.append(calc_acc(train_pred, Ttrain))
        test_accuracy.append(calc_acc(test_pred, Ttest))
                                               
        # [epoch end]
    
    print_accuracy_diff("SGDlinear", train_accuracy, test_accuracy)
    plot_figures("Question 2(d)", W, train_avg_ce, test_avg_ce, train_accuracy, test_accuracy)
    

print("\nQ2(d):")
#SGDlinear(500, 30, 2, 0.1, 200)


# QUESTION 3----------------------------------------
print("\nQuestion 3")
print("----------")

# Q3(a) ~~~~~~~~~
print("\nQ3(a):")
qda = da.QuadraticDiscriminantAnalysis(store_covariance=True)
qda.fit(Xtrain, Ttrain) # train the model
print("Training Data Accuracy: ", qda.score(Xtrain, Ttrain))
print("Test Data Accuracy: ", qda.score(Xtest, Ttest))
plot_dboundaries(qda, "Question 3(a): decision boundaries for QDA")

# Q3(b) ~~~~~~~~~
print("\nQ3(b):")
nb = naiv.GaussianNB()
nb.fit(Xtrain, Ttrain) # train the model
print("Training Data Accuracy: ", nb.score(Xtrain, Ttrain))
print("Test Data Accuracy: ", nb.score(Xtest, Ttest))
plot_dboundaries(nb, "Question 3(b): decision boundaries for naive bayes")

# Q3(f) ~~~~~~~~~
def EstMean(X, T):
    """
    Return the mean vector for each class in QDA
    X: data matrix of input vectors
    T: one-hot matrix of class labels
    """
    N = np.sum(T, axis=0)
    return T.T @ X / N.reshape(N.size, 1)


print("\nQ3(f):")
Ttrain_hot = one_hot(Ttrain)
mu = EstMean(Xtrain, Ttrain_hot)
mu_sk = qda.means_
tsd = np.sum((mu-mu_sk) ** 2)
print("Total Squared Difference: ", tsd)

# Q3(g) ~~~~~~~~~
def EstCov(X, T):
    """
    Return the covariance matrix for each class in QDA
    X: data matrix of input vectors
    T: one-hot matrix of class labels
    """
    mu = EstMean(X, T)
    n = Xtrain.shape[0]
    i = Xtrain.shape[1]
    k = mu.shape[0]
    N = np.sum(T, axis=0) - 1
    
    A = Xtrain.reshape(n, 1, i) - mu.reshape(1, k, i)
    B = A.reshape(n, k, i, 1) * A.reshape(n, k, 1, i)
    C = T.reshape(n, k, 1, 1) * B
    D = np.sum(C, axis=0)
    return D / N.reshape(k, 1, 1)


print("\nQ3(g):")
sig = EstCov(Xtrain, Ttrain_hot)
sig_sk = qda.covariance_
tsd = np.sum((sig-sig_sk) ** 2)
print("Total Squared Difference: ", tsd)

# Q3(h) ~~~~~~~~~
def EstPrior(T):
    """
    Return the prior probability of each class in QDA
    T: one-hot matrix of class labels
    """
    return np.sum(T, axis=0) / T.shape[0]
    

print("\nQ3(h):")
pp = EstPrior(Ttrain_hot)
pp_sk = qda.priors_
tsd = np.sum((pp-pp_sk) ** 2)
print("Total Squared Difference: ", tsd)

# Q3(i) ~~~~~~~~~
def EstPost(mean, cov, prior, X):
    k = mean.shape[0]
    p = []
    
    for c in range(k):
        p.append(sci.multivariate_normal.pdf(X, mean[c], cov[c]))
    
    parr = np.array(p)
    px = parr.T @ prior.reshape(prior.shape[0], 1)
    post = parr.T * prior / px
    return post


print("\nQ3(i):")
post = EstPost(mu, sig, pp, Xtest)
post_sk = qda.predict_proba(Xtest)
tsd = np.sum((post-post_sk) ** 2)
print("Total Squared Difference: ", tsd)

# Q3(j) ~~~~~~~~~
print("\nQ3(j):")
Ttest_hot = one_hot(Ttest)
my_lce = calc_lce(post, Ttest_hot)
sk_lce = calc_lce(post_sk, Ttest_hot)
print("Avg. Cross Entropy (QDA): ", my_lce)
print("Avg. Cross Entropy (sklearn's QDA): ", sk_lce)
print("Difference (QDA-sklearn's QDA): ", my_lce - sk_lce)

# Q3(k) ~~~~~~~~~
print("\nQ3(k):")
my_pred = np.argmax(post, axis=1)
my_score = calc_acc(my_pred, Ttest)
sk_score = qda.score(Xtest, Ttest)
print("Test Accuracy (QDA): ", my_score)
print("Test Accuracy (sklearn's QDA): ", sk_score)
print("Difference: ", my_score - sk_score)


# QUESTION 4----------------------------------------
print("\nQuestion 4")
print("----------")

# Q4(a) ~~~~~~~~~
print("\nQ4(a):")
def mplc_init(nunit):
    np.random.seed(7)
    return nn.MLPClassifier(hidden_layer_sizes=nunit,
                           activation='logistic',
                           solver='sgd',
                           learning_rate_init=0.01,
                           max_iter=10000,
                           tol=1e-6)


def print_data_acc(net, nunit):
    print("\n{} Hidden Unit".format(nunit))
    print("Training Data Accuracy: ", net.score(Xtrain, Ttrain))
    print("Test Data Accuracy: ", net.score(Xtest, Ttest))


net5 = mplc_init(5)
net5.fit(Xtrain, Ttrain)
print_data_acc(net5, 5)
plot_dboundaries(net5, "Question 4(a): neural net with 5 hidden units")

# Q4(b) ~~~~~~~~~
def subplot_dboundaries(clf, title):
    bl2d.plot_data(Xtrain, Ttrain)
    bl2d.boundaries(clf)
    plt.title(title)
    
    
# Define neural net classifiers 
net1 = mplc_init(1)
net2 = mplc_init(2)
net4 = mplc_init(4)
net10 = mplc_init(10)

# train clf
net1.fit(Xtrain, Ttrain)
net2.fit(Xtrain, Ttrain)
net4.fit(Xtrain, Ttrain)
net10.fit(Xtrain, Ttrain)

# training and test accuracies
print("\nQ4(b):")
print_data_acc(net1, 1)
print_data_acc(net2, 2)
print_data_acc(net4, 4)
print_data_acc(net10, 10)

# plot decision boundaries
plt.subplot(2,2,1)
subplot_dboundaries(net1, "1 Hidden Unit")

plt.subplot(2,2,2)
subplot_dboundaries(net2, "2 Hidden Unit")

plt.subplot(2,2,3)
subplot_dboundaries(net4, "4 Hidden Unit")

plt.subplot(2,2,4)
subplot_dboundaries(net10, "10 Hidden Unit")

plt.suptitle("Question 4(b): Neural net decision boundaries")

plt.show()

