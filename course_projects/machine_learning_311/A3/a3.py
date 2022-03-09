#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSC311H5F - Assignment 3
@author: Josh Alexander
UTORid: alexa364
Student Number: 1005434458
"""

import pickle
import bonnerlib2D as bl2d
import numpy as np
import matplotlib.pyplot as plt
import sklearn.utils as utl

import sklearn.neural_network as nn
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# HELPER FUNCTIONS
def calc_acc(prob, t):
    '''
    Calculate the accuracy of probability dist. <prob> given <t>
    '''
    pred = np.argmax(prob, axis=1)
    return np.sum(np.array(pred == t), dtype=int)/pred.size

# RETRIEVE DATA
with open('mnistTVT.pickle', 'rb') as f:
    Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = pickle.load(f)

# ==========    
# QUESTION 1
# ==========

print("\nQuestion 1")

# Q1(a) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Xtrain_small = Xtrain[:500,:]
Ttrain_small = Ttrain[:500]

# Q1(b) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nQ1(b):")
np.random.seed(7)
net5 = nn.MLPClassifier(hidden_layer_sizes=5,
                          activation='logistic',
                          solver='sgd',
                          alpha=0,
                          learning_rate_init=0.1,
                          max_iter=10000)

net5.fit(Xtrain_small, Ttrain_small)
net5_val_acc = net5.score(Xval, Tval)
print("NN5 Training Acc. = ", net5.score(Xtrain_small, Ttrain_small))
print("NN5 Validation Acc. = ", net5_val_acc)

# Q1(d) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nQ1(d):")
pred_val_acc = calc_acc(net5.predict_proba(Xval), Tval)
print("Difference = ", net5_val_acc - pred_val_acc)

# Q1(e) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
np.random.seed(7)
prob = None
for c in range(7):
    x, t = utl.resample(Xtrain_small, Ttrain_small, n_samples=500)
    net5.fit(x, t)
    net5_prob = net5.predict_proba(Xval)
    if c == 0:
        prob = net5_prob
    else:
        prob += net5_prob

avg_prob = prob/7
pred_val_acc = calc_acc(avg_prob, Tval)

# Q1(g) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
np.random.seed(7)
prob = None
acc = []

for i in range(100):
    x, t = utl.resample(Xtrain_small, Ttrain_small, n_samples=500)
    net5.fit(x, t)
    net5_prob = net5.predict_proba(Xval)
    if i == 0:
        prob = net5_prob
    else:
        prob += net5_prob
    avg_prob = prob/(i+1)
    acc.append(calc_acc(avg_prob, Tval))
    
plt.plot(acc)
plt.xlabel("Iteration Number")
plt.ylabel("Accuracy")
plt.title("Question 1(g)")
plt.show()

# ==========    
# QUESTION 2
# ==========
print("\nQuestion 2")

# Environment Detail
GRIDSIZE = 10
STARTPOS = [5, 3]

# Action space
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
ACTIONSPACE = [UP, RIGHT, DOWN, LEFT]

# Objects
BARRIER = -1
EMPTY = 0
GOAL = 1
AGENT = 2

# Q-function
# [0] move UP policy
# [1] move RIGHT policy
# [2] move DOWN policy
# [3] move LEFT policy
Qf = np.array([])

# Q2(a) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Environment Grid 10x10, 
env = np.zeros((GRIDSIZE, GRIDSIZE), dtype=int)

# Adding barriers
env[3, 1:3] = BARRIER
env[7, 2:6] = BARRIER
env[2:7, 5] = BARRIER
env[4, 5:9] = BARRIER
env[7, 8] = BARRIER

# Adding player and goal
env[5, 3] = AGENT
env[5, 6] = GOAL

plt.imshow(env)
plt.title("Question 2(a): Grid World")
plt.show()

# Q2(b) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def possible_moves(L):
    '''
    Return all the possible moves given L.
    Return -1 if L is not valid.
    
    L -> location of the agent
    '''
    x, y = L
    pmoves = []
    
    # Input validity check
    if x < 0 or x > GRIDSIZE - 1 or \
        y < 0 or y > GRIDSIZE - 1:
            return -1
    
    # Out-of-bound and barrier check 
    if x < GRIDSIZE - 1 and env[x + 1, y] != BARRIER:
        pmoves.append(DOWN)
    if x > 0 and env[x - 1, y] != BARRIER:
        pmoves.append(UP)
    
    if y < GRIDSIZE - 1 and env[x, y + 1] != BARRIER:
        pmoves.append(RIGHT)
    if y > 0 and env[x, y - 1] != BARRIER:
        pmoves.append(LEFT)
    
    return pmoves
        

def Trans(L, a):
    '''
    Return the new location of the agent and the immediate reward
    Return <L> if <a> is not a valid move.
    
    L -> location of the agent
    a -> action
    '''
    x, y = L
    pmoves = possible_moves(L)
    
    # Action not valid
    if pmoves == -1 or a not in pmoves:
        return L, 0
    
    # Move coordinate
    if a == UP:
        x -= 1
    elif a == DOWN:
        x += 1
    elif a == LEFT:
        y -= 1
    elif a == RIGHT:
        y += 1
    
    # Check immediate reward
    if env[x, y] == GOAL:
        return [x, y], 25
    else:
        return [x, y], 0


# Q2(c) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def choose(L, beta):
    '''
    Softmax Exploration Policy
    Select an action probabilistically and return it.
    
    L -> location of the agent
    beta -> softmax parameter
    '''
    x, y = L
    Qup = Qf[UP, x, y]
    Qright = Qf[RIGHT, x, y]
    Qdown = Qf[DOWN, x, y]
    Qleft = Qf[LEFT, x, y]
    
    Q = np.exp(np.array([Qup, Qright, Qdown, Qleft]) * beta)
    denom = np.sum(Q)

    prob = Q/denom
    return np.random.choice(ACTIONSPACE, 1, p=prob)[0]


# Q2(d) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def updateQ(L, a, alpha, gamma):
    '''
    Updates Qf and return the new L.
    
    L -> location of the agent
    a -> action
    alpha -> learning rate
    gamma -> discount factor
    '''
    global Qf
    
    # Take action, get R and new L
    x, y = L
    L2, R = Trans(L, a)
    x2, y2 = L2
    Qmax = np.max(Qf[:, x2, y2])
    update = alpha*(R + gamma*Qmax - Qf[a, x, y])
    
    # Update Qf
    Qf[a, x, y] += update

    return L2


# Q2(e) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def episode(L, alpha, gamma, beta):
    '''
    Move the agent from <L> to the goal state
    with softmax exploration policy.
    
    Return the number of iteration needed to arrive
    at the goal.
    
    L -> location of the agent
    alpha -> learning rate
    gamma -> discount factor
    beta -> softmax parameter
    '''
    x, y = L
    cell = env[x, y]
    move = 0

    while cell != GOAL:
        a = choose([x, y], beta) # choose best action
        L2 = updateQ([x, y], a, alpha, gamma) # take action and update Qf
        x, y = L2
        cell = env[x, y]
        move += 1
    
    return move


# Q2(f) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def learn(N, L, alpha, gamma, beta):
    '''
    Initialize Q-function and performs N episodes of Q-learning
    that start from L.
    
    Return the number of iteration needed to complete
    each episode.

    N -> number of episodes
    L -> starting location of the agent
    alpha -> learning rate
    gamma -> discount factor
    beta -> softmax parameter
    '''
    moves = []
    
    # Initialize Q
    global Qf
    Qf = np.zeros((4, GRIDSIZE, GRIDSIZE))
    
    # Loop N episodes
    for _ in range(N):
        move = episode(L, alpha, gamma, beta)
        moves.append(move)
    
    return moves


# Q2(g) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
np.random.seed(7)
moves = learn(50, STARTPOS, 1, 0.9, 1)
plt.plot(moves)
plt.xlabel("Episode")
plt.ylabel("Episode Length")
plt.title("Question 2(g): one run of Q learning")
plt.grid()
plt.show()

# Q2(h) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
np.random.seed(7)
moves = learn(50, STARTPOS, 1, 0.9, 0)
plt.plot(moves)
plt.xlabel("Episode")
plt.ylabel("Episode Length")
plt.title("Question 2(h): one run of Q learning (beta=0)")
plt.grid()
plt.show()

# Q2(i) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
np.random.seed(7)
avg_moves = np.zeros(50)
for _ in range(100):
    moves = learn(50, STARTPOS, 1, 0.9, 1)
    avg_moves += moves
    
avg_moves = avg_moves/100
plt.plot(avg_moves)
plt.xlabel("Episode")
plt.ylabel("Avg. Episode Length")
plt.title("Question 2(i): 100 runs of Q learning")
plt.xlim(0, 50)
plt.ylim(0, 1000)
plt.grid()
plt.show()

# Q2(j) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
np.random.seed(7)
avg_moves = np.zeros(50)
for _ in range(100):
    moves = learn(50, STARTPOS, 1, 0.9, 0.1)
    avg_moves += moves
    
avg_moves = avg_moves/100
plt.plot(avg_moves)
plt.xlabel("Episode")
plt.ylabel("Avg. Episode Length")
plt.title("Question 2(j): 100 runs of Q learning (beta=0.1)")
plt.xlim(0, 50)
plt.ylim(0, 1000)
plt.grid()
plt.show()

# Q2(k) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
np.random.seed(7)
avg_moves = np.zeros(50)
for _ in range(100):
    moves = learn(50, STARTPOS, 1, 0.9, 0.01)
    avg_moves += moves
    
avg_moves = avg_moves/100
plt.plot(avg_moves)
plt.xlabel("Episode")
plt.ylabel("Avg. Episode Length")
plt.title("Question 2(k): 100 runs of Q learning (beta=0.01)")
plt.xlim(0, 50)
plt.ylim(0, 1000)
plt.grid()
plt.show()

# Q2(m) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
np.random.seed(7)
learn(50, STARTPOS, 1, 0.9, 1)
Qmax = np.max(Qf, axis=0)
plt.imshow(Qmax)
plt.title("Question 2(m): Qmax for beta=1")
plt.show()

# Q2(o) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
np.random.seed(7)
for c in range(9):
    learn(50, STARTPOS, 1, 0.9, 1)
    Qmax = np.max(Qf, axis=0)
    plt.subplot(3, 3, c + 1)
    plt.axis(False)
    plt.imshow(Qmax)

plt.suptitle("Question 2(o): Qmax for beta=1")
plt.show()

# Q2(p) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
np.random.seed(7)
for c in range(9):
    learn(50, STARTPOS, 1, 0.9, 0)
    Qmax = np.max(Qf, axis=0)
    plt.subplot(3, 3, c + 1)
    plt.axis(False)
    plt.imshow(Qmax)
    
plt.suptitle("Question 2(p): Qmax for beta=0")
plt.show()

# Q2(r) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nQ2(r):")

def path(L):
    '''
    Use greedy policy to find an optimal path from
    location <L> to the goal state.
    
    Return the number of iteration needed to arrive
    at the goal and the path taken.
    
    L -> location of the agent
    '''
    x, y = L
    p = np.copy(env)
    cell = env[x, y]
    c = 0

    while cell != GOAL:
        a = np.argmax(Qf[:, x, y])
        L2, R = Trans([x, y], a)
        c += 1 # action taken
        x, y = L2
        p[x, y] = AGENT # represent cell crossed by the agent
        cell = env[x, y]
        
    return c, p
    

np.random.seed(7)
learn(50, STARTPOS, 1, 0.9, 1)
c, p = path(STARTPOS)
plt.imshow(p)
plt.title("Question 2(r): an optimal path, beta=1")
plt.show()

print("Path length: ", c)

# Q2(s) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nQ2(s):")
np.random.seed(7)
avg_c = 0
for i in range(9):
    learn(50, STARTPOS, 1, 0.9, 1)
    c, p = path(STARTPOS)
    avg_c += c
    plt.subplot(3, 3, i + 1)
    plt.axis(False)
    plt.imshow(p)
 
avg_c /= 9  
plt.suptitle("Question 2(s): optimal paths, beta=1")
plt.show()

print("Average path length: ", avg_c)

# Q2(t) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nQ2(t):")
np.random.seed(7)
avg_c = 0
for i in range(9):
    learn(50, STARTPOS, 1, 0.9, 0)
    c, p = path(STARTPOS)
    avg_c += c
    plt.subplot(3, 3, i + 1)
    plt.axis(False)
    plt.imshow(p)

avg_c /= 9
plt.suptitle("Question 2(t): optimal paths, beta=0")
plt.show()

print("Average path length: ", avg_c)

# ==========    
# QUESTION 3
# ==========
print("\nQuestion 3")

# RETRIEVE DATA
with open('cluster_data.pickle', 'rb') as file:
    dataTrain, dataTest = pickle.load(file)
    
X2train, _ = dataTrain
X2test, _ = dataTest

# HELPER FUNCTION FROM A2
def one_hot(Tint):
    '''
    Converts integer target values to one-hot encodings
    In dim: Tint -> [N]
    Out dim: Thot -> [N, J], where J is the number of classes
    '''
    N = Tint.size
    Tint = Tint.reshape((N,1))
    C = np.arange(0, np.max(Tint)+1)
    return np.array((Tint == C), dtype=int)


# Q3(a) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
<<<<<<< Updated upstream
cluster = clt.KMeans(n_clusters=3)
cluster.fit(Xtrain)
=======
print("\nQ3(a):")
kmeans = KMeans(n_clusters=3).fit(X2train)
pred = kmeans.predict(X2train)
R = one_hot(pred)
centers = kmeans.cluster_centers_

bl2d.plot_clusters(X2train, R)
plt.scatter(centers[:, 0], centers[:, 1], c=['#000000'])
plt.title("Question 3(a): K means")
plt.show()

print("Training score: ", kmeans.score(X2train))
print("Test score: ", kmeans.score(X2test))

# Q3(b) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nQ3(b):")
gmix = GaussianMixture(n_components=3, 
                       covariance_type='diag', 
                       tol=1e-7).fit(X2train)
R = gmix.predict_proba(X2train)
centers = gmix.means_

bl2d.plot_clusters(X2train, R)
plt.scatter(centers[:, 0], centers[:, 1], c=['#000000'])
plt.title("Question 3(b): Gaussian mixture model (diagonal)")
plt.show()

train_diag_score = gmix.score(X2train)
test_diag_score = gmix.score(X2test)

print("Training score: ", train_diag_score)
print("Test score: ", test_diag_score)

# Q3(c) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nQ3(c):")
gmix = GaussianMixture(n_components=3,
                       covariance_type='full',
                       tol=1e-7).fit(X2train)
R = gmix.predict_proba(X2train)
centers = gmix.means_

bl2d.plot_clusters(X2train, R)
plt.scatter(centers[:, 0], centers[:, 1], c=['#000000'])
plt.title("Question 3(c): Gaussian mixture model (full)")
plt.show()
>>>>>>> Stashed changes

train_full_score = gmix.score(X2train)
test_full_score = gmix.score(X2test)

print("Training score: ", train_full_score)
print("Test score: ", test_full_score)
print("Q3c-Q3b Test scores = ", test_full_score - test_diag_score)

# Q3(e) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nQ3(e):")

def gmm(X, K, I):
    # I don't know
    pass


def scoreGMM(X, MU, Sigma, Pi):
    # I don't know
    pass


print("I don't know")

# Q3(h) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nQ3(h):")
gmix = GaussianMixture(n_components=10,
                       covariance_type='diag',
                       tol=1e-3).fit(Xtrain)
mean_vec = gmix.means_

for i in range(10):
    plt.subplot(4, 3, i + 1)
    plt.axis(False)
    plt.imshow(mean_vec[i, :].reshape(28, 28))

plt.suptitle("Question 3(h): mean vectors for 50,000 MNIST training points")
plt.show()

train_score = gmix.score(Xtrain)
test_score = gmix.score(Xtest)
print("Training score: ", train_score)
print("Test score: ", test_score)

# Q3(i) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nQ3(i):")
gmix = GaussianMixture(n_components=10,
                       covariance_type='diag',
                       tol=1e-3).fit(Xtrain_small)
mean_vec = gmix.means_

for i in range(10):
    plt.subplot(4, 3, i + 1)
    plt.axis(False)
    plt.imshow(mean_vec[i, :].reshape(28, 28))

plt.suptitle("Question 3(i): mean vectors for 500 MNIST training points")
plt.show()

train_score = gmix.score(Xtrain_small)
test_score = gmix.score(Xtest)
print("Training score: ", train_score)
print("Test score: ", test_score)

# Q3(j) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nQ3(j):")
Xtrain_10 = Xtrain[:10, :]
gmix = GaussianMixture(n_components=10,
                       covariance_type='diag',
                       tol=1e-3).fit(Xtrain_10) 
mean_vec = gmix.means_

for i in range(10):
    plt.subplot(4, 3, i + 1)
    plt.axis(False)
    plt.imshow(mean_vec[i, :].reshape(28, 28))

plt.suptitle("Question 3(j): mean vectors for 10 MNIST training points")
plt.show()

train_score = gmix.score(Xtrain_10)
test_score = gmix.score(Xtest)
print("Training score: ", train_score)
print("Test score: ", test_score)

