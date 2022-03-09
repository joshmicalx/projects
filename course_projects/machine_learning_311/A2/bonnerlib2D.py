#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 18:01:18 2017

@author: anthonybonner
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D




# generate a scatter plot of cluster data.
# X is a data matrix, and T is a vector of target values.
# assume three classes (target values).
def plot_data(X,T):
    colors = np.array(['r','b','g'])    # red for class 0 , blue for class 1, green for class 2
    plt.scatter(X[:, 0], X[:, 1], color=colors[T],s=2)
    xmin,ymin = np.min(X,axis=0) - 0.1
    xmax,ymax = np.max(X,axis=0) + 0.1
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)




# Plot the decision boundaries of classifier clf
# and colour the decision regions.
# Assume three classes.
def boundaries(clf):
    
    # The extent of xy space
    ax = plt.gca()
    x_min,x_max = ax.get_xlim()
    y_min,y_max = ax.get_ylim()
    
    # form a mesh/grid over xy space
    h = 0.02    # mesh granularity
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(),yy.ravel()]
    
    # evaluate the decision function at the grid points
    U = clf.predict(mesh)    ###### Modify this line ######
    
    # Use pale colours for the decision regions.
    U = U.reshape(xx.shape)
    mylevels=[-0.5,0.5,1.5,2.5]
    ax.contourf(xx, yy, U, levels=mylevels, colors=('red','blue','green'), alpha=0.2)
    
    # draw the decision boundaries in solid black
    ax.contour(xx, yy, U, levels=3, colors='k', linestyles='solid')
    



# Plot the decision boundaries of classifier clf
# and colour the decision regions.
# Assume three classes.
# W is a weight matrix
# b is a bias vector.
# predict is a function, where predict(X,W,b) returns the predicted class
#    of every row vector in data matrix X
def boundaries2(W,b,predict):
    
    # The extent of xy space
    ax = plt.gca()
    x_min,x_max = ax.get_xlim()
    y_min,y_max = ax.get_ylim()
    
    # form a mesh/grid over xy space
    h = 0.02    # mesh granularity
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(),yy.ravel()]    # shape = [N,K]
    
    # evaluate the decision function at the grid points
    U = predict(mesh,W,b)
    
    # Use pale colours for the decision regions.
    U = U.reshape(xx.shape)
    mylevels=[-0.5,0.5,1.5,2.5]
    ax.contourf(xx, yy, U, levels=mylevels, colors=('red','blue','green'), alpha=0.2)
    
    # draw the decision boundaries in solid black
    ax.contour(xx, yy, U, levels=3, colors='k', linestyles='solid')




# Plot the decision function of classifier clf in 3D.
# if Cflag=1 (the default), draw a contour plot of the decision function
# beneath the 3D plot.

def df3D(clf,cFlag=1):    
    
    # the extent of xy space
    ax = plt.gca()
    x_min,x_max = ax.get_xlim()
    y_min,y_max = ax.get_ylim()
    
    # form a mesh/grid over xy space
    h = 0.01    # mesh granularity
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(),yy.ravel()]
    
    # evaluate the decision function at the grid points
    Z = clf.predict_proba(mesh)[:,1]
    
    # plot the contours of the decision function
    Z = Z.reshape(xx.shape)
    ax.plot_surface(xx, yy, Z, cmap=cm.RdBu, linewidth=0, rcount=75, ccount=75)
    mylevels=np.linspace(0.0,1.0,11)
    ax.contour(xx,yy,Z,levels=mylevels,linestyles='solid',linewidths=3,cmap=cm.RdBu)
    
    # limits of the vertical axis
    z_min = 0.0
    z_max = 1.0
    
    if cFlag == 1:
        # display a contour plot of the decision function
        z_min = -0.5
        ax.contourf(xx, yy, Z, levels=mylevels, cmap=cm.RdBu, offset=z_min)
    
    ax.set_zlim(z_min,z_max)
    