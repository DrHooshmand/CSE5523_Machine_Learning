# EM.py
# --------------
# Licensing Information:  You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to The Ohio State University, including a link to http://aritter.github.io/courses/5523_fall18.html
#
# Attribution Information: This assignment was developed at The Ohio State University
# by Alan Ritter (ritter.1492@osu.edu).

import random
import math
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

#GLOBALS/Constants
VAR_INIT = 1

def logExpSum(x):
    #TODO: implement logExpSum
    xmax = max(x)
    dum = [X-xmax for X in x]
    #
    # return xmax + math.log( sum (math.exp( dum ) ) )
    return xmax + np.log(sum( np.exp(dum)   ) )


def readTrue(filename='wine-true.data'):
    f = open(filename)
    labels = []
    splitRe = re.compile(r"\s")
    for line in f:
        labels.append(int(splitRe.split(line)[0]))
    return labels

#########################################################################
#Reads and manages data in appropriate format
#########################################################################
class Data:
    def __init__(self, filename):
        self.data = []
        f = open(filename)
        (self.nRows,self.nCols) = [int(x) for x in f.readline().split(" ")]
        for line in f:
            self.data.append([float(x) for x in line.split(" ")])

    #Computers the range of each column (returns a list of min-max tuples)
    def Range(self):
        ranges = []
        for j in range(self.nCols):
            min = self.data[0][j]
            max = self.data[0][j]
            for i in range(1,self.nRows):
                if self.data[i][j] > max:
                    max = self.data[i][j]
                if self.data[i][j] < min:
                    min = self.data[i][j]
            ranges.append((min,max))
        return ranges

    def __getitem__(self,row):
        return self.data[row]

#########################################################################
#Computes EM on a given data set, using the specified number of clusters
#self.parameters is a tuple containing the mean and variance for each gaussian
#########################################################################
class EM:
    def __init__(self, data, nClusters,rrr):
        #Initialize parameters randomly...
        random.seed(rrr)
        self.thresh = 1e-8
        self.ws = np.zeros( (nClusters,data.nRows) )
        self.parameters = []
        self.priors = []        #Cluster priors
        self.nClusters = nClusters
        self.data = data
        ranges = data.Range()
        for i in range(nClusters):
            p = []
            initRow = random.randint(0,data.nRows-1)
            for j in range(data.nCols):
                #Randomly initalize variance in range of data
                p.append((random.uniform(ranges[j][0], ranges[j][1]), VAR_INIT*(ranges[j][1] - ranges[j][0])))
            self.parameters.append(p)

        #Initialize priors uniformly
        for c in range(nClusters):
            self.priors.append(1/float(nClusters))

    def LogLikelihood(self, data):
        logLikelihood = 0.0
        k = self.nClusters
        rows = data.nRows

        for i in range(rows):
            dum =[]
            for j in range(k):
                # considering priors
                    dum.append(self.priors [j] * self.LogProb(i,j,data))
                # not considering priors
                    #dum.append( self.LogProb(i, j, data))
            dum_inter = logExpSum(dum)
            logLikelihood +=dum_inter
        return logLikelihood

    def LLL(self,data):
        # TODO: For separating the loglikelihood of each cluster in each data point
        # Implemented for separating the loglikelihood of each clusters within each data point. This is helpful for part (c)
        k = self.nClusters
        rows = data.nRows
        LLL_each_data = np.zeros((rows, k) )

        for i in range(rows):
            for j in range(k):
                LLL_each_data[i][j] = self.priors[j] * self.LogProb(i, j, data)
        return LLL_each_data


    def Estep(self):
        #TODO: E-step
        n = self.data.nRows
        p = self.data.nCols
        k = self.nClusters

        for i in range(k):
            for  j in range(n):
                self.ws[i,j] = self.LogProb(j, i, self.data)
        # For Calculating the normalization factor which is logexpsum of loglikelihoods over the clusters for each data
        normalized = []
        for i in range(n):
            normalized.append(logExpSum(self.ws[:,i]))

        for i in range(k):
            for j in range(n):
                self.ws[i, j] = np.exp(self.ws[i, j] )/ np.exp( normalized[j])
        # ws /= ws.sum(0)
        return self.ws



    #Update the parameter estimates
    def Mstep(self):
        #TODO: M-step
        k = self.nClusters
        rows = self.data.nRows
        cols = self.data.nCols

        for i in range(k):
            new_mios = np.zeros(cols)
            new_vars = np.zeros(cols)
            sum_probs = 0
            mean_clus = [x for (x,y) in e.parameters[i]]
            var_clus =  [y for (x,y) in e.parameters[i]]

            for j in range(rows):
                plus = np.multiply(self.data[j], self.ws[i][j])
                new_mios += plus

                subt = np.subtract(self.data[j],mean_clus)
                plus2 = np.multiply( np.multiply(subt, self.ws[i][j]) , subt)
                new_vars += plus2

                sum_probs += self.ws[i][j]

            if sum_probs == 0:
                sum_probs = self.thresh


            new_mios = new_mios / sum_probs
            new_vars = new_vars /sum_probs
            self.priors[i] = sum_probs /rows

            dum=[]
            for l in range(cols):
                dum.append( (new_mios[l], new_vars[l]) )
            self.parameters[i] = dum



    #Computes the probability that row was generated by cluster
    def LogProb(self, row, cluster, data):
        #TODO: compute probability row i was generated by cluster k
        Lprob = 0
        x = data[row]
        k = self.nClusters
        col = data.nCols
        Lprob += -0.5 * col * np.log (2*np.pi)

        for i in range(col):
            mio = self.parameters[cluster][i][0]
            sigma = self.parameters[cluster][i][1]
            if sigma == 0 :
                sigma = self.thresh
            Lprob += (-0.5 * np.log(sigma ) ) - 0.5 * ( (x[i] - mio) **2 ) / sigma

        Lprob += np.log(self.priors[cluster])
        return Lprob




    def Run(self, maxsteps=100, testData=None):
        # TODO: Implement EM algorithm
        trainLikelihood = 0.0
        testLikelihood = 0.0
        if testData!=None:
            self.data = testData

        old_ll = 0
        new_ll = 0
        change = 10
        s = 1

        results_train=[]
        results_test = []
        #for s in range(maxsteps):
        print("Running Begins \n")
        while s<=maxsteps or change > 0.01:
            if testData!=None:
                self.Estep()
            else:
                self.Estep()
                self.Mstep()
            new_ll = self.LogLikelihood(self.data)
            change =  (new_ll - old_ll)/old_ll *100
            results_train.append( (s,new_ll,change) )
            old_ll = new_ll
            print("step:", s)
            s+=1
            if np.isinf(change):
                change =1e80

        # if testData!=None:
        #     self.data = testData
        #     self.Run()
        #     results_test = results_train
        # return results_train, results_test

        ##for part (a)
        #return results_train
        ##for part (b)-(d)
        return new_ll




    def pll(self,results,name="train"):
        # implemented for plotting purposes
        plt.figure()
        s = [list(s) for s in zip(*results)][0]
        ll = [list(s) for s in zip(*results)][1]
        change = [list(s) for s in zip(*results)][2]
        plt.scatter(s, ll)
        plt.xlabel("steps")
        plt.ylabel("Likelihood")
        plt.savefig(name+"_ll"+".pdf")

        plt.figure()
        plt.scatter(s, change)
        plt.xlabel("steps")
        plt.ylabel("Change in LogLiklihood")
        plt.axhline(y=0.1, color='r', linestyle='-')
        plt.axhline(y=-0.1, color='r', linestyle='-')
        plt.savefig(name + "_change"+".pdf")



if __name__ == "__main__":
    # d = Data('wine.train')
    # n_clus = 10
    # if len(sys.argv) > 1:
    #     e = EM(d, int(sys.argv[1]))
    # else:
    #     e = EM(d, 3)
    # e.Run(100)

    d = Data('wine.train')
    d_test = Data('wine.test')

    # if len(sys.argv) > 1:
    #     e = EM(d, int(sys.argv[1]))
    # else:
    #     e = EM(d, 3)
    #e.Run(200,testData=d_test)
    #results_train, results_test=e.Run(40,testData=d_test)

# Uncomment for Section (a)
#     ####################################################
#     e = EM(d, 3,70)
#     results_train = e.Run(30)
#     e.pll(results_train, name="train")
#
#     results_test = e.Run(30,testData=d_test)
#     e.pll(results_test, name="test")
#     ####################################################

#########################################################################
# # Uncomment for Seed test for part b
#     seed_train =[]
#     seed_test = []
#     for i in range(10):
#         e = EM(d, 3,i)
#         #random.seed(i)
#         results_train = e.Run(100)
#         results_test = e.Run(100, testData=d_test)
#         seed_train.append(results_train)
#         seed_test.append (results_test)
#
#     plt.figure()
#     plt.scatter (range(10),seed_train)
#     plt.xlabel("seed")
#     plt.ylabel("loglikelihood")
#     plt.savefig("train_seed.pdf")
#
#     plt.figure()
#     plt.scatter (range(10),seed_test)
#     plt.xlabel("seed")
#     plt.ylabel("loglikelihood")
#     plt.savefig("test_seed.pdf")

##########################################################################
    # # Uncomment for : Implemented for Part C

    data_true = np.loadtxt("wine-true.data")
    true_label = list(data_true[:, 0])

    accuracy_results= []
    for s in range(10):

        e = EM(d, 3, s)
        e.Run(100)
        LL_separate = e.LLL(d)

        trained_label =[]
        for i in range(d.nRows):
           trained_label.append(LL_separate[i].argmax() +1)


        check = []
        for i in range(d.nRows):
            if true_label[i] == trained_label[i]:
                check.append(1)
            else:
                check.append(0)

        accuracy = sum (check) / len(check) *100
        accuracy_results.append(accuracy)

    plt.figure()
    plt.scatter (range(10),accuracy_results)
    plt.xlabel("seed")
    plt.ylabel("Accuracy")
    plt.savefig("accuracy_seed.pdf")

    ##########################################################################
# # Uncomment for: Implemented for part (d)
#
#     clus_train =[]
#     clus_test = []
#     for i in range(1,11):
#         e = EM(d, i,2)
#         #random.seed(i)
#         results_train = e.Run(100)
#         results_test = e.Run(100, testData=d_test)
#         clus_train.append(results_train)
#         clus_test.append (results_test)
#
#     plt.figure()
#     plt.scatter (range(1,11),clus_train)
#     plt.xlabel("Clusters")
#     plt.ylabel("loglikelihood")
#     plt.savefig("train_cluster.pdf")
#
#     plt.figure()
#     plt.scatter (range(1,11),clus_test)
#     plt.xlabel("Clusters")
#     plt.ylabel("loglikelihood")
#     plt.savefig("test_cluster.pdf")



print("done")
