# linear_brain.py
# --------------
# Licensing Information:  You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to The Ohio State University, including a link to http://aritter.github.io/courses/5523_fall18.html
#
# Attribution Information: This assignment was developed at The Ohio State University
# by Alan Ritter (ritter.1492@osu.edu).

import scipy.io as sio
import numpy as np
from random import randint
import matplotlib.pyplot as plt
import matplotlib
from itertools import cycle
from collections import namedtuple


starplus = sio.loadmat("data-starplus-04847-v7.mat")

###########################################
metadata = starplus['meta'][0,0]
#meta.study gives the name of the fMRI study
#meta.subject gives the identifier for the human subject
#meta.ntrials gives the number of trials in this dataset
#meta.nsnapshots gives the total number of images in the dataset
#meta.nvoxels gives the number of voxels (3D pixels) in each image
#meta.dimx gives the maximum x coordinate in the brain image. The minimum x coordinate is x=1. meta.dimy and meta.dimz give the same information for the y and z coordinates.
#meta.colToCoord(v,:) gives the geometric coordinate (x,y,z) of the voxel corresponding to column v in the data
#meta.coordToCol(x,y,z) gives the column index (within the data) of the voxel whose coordinate is (x,y,z)
#meta.rois is a struct array defining a few dozen anatomically defined Regions Of Interest (ROIs) in the brain. Each element of the struct array defines on of the ROIs, and has three fields: "name" which gives the ROI name (e.g., 'LIFG'), "coords" which gives the xyz coordinates of each voxel in that ROI, and "columns" which gives the column index of each voxel in that ROI.
#meta.colToROI{v} gives the ROI of the voxel corresponding to column v in the data.
study      = metadata['study']
subject    = metadata['subject']
ntrials    = metadata['ntrials'][0][0]
nsnapshots = metadata['nsnapshots'][0][0]
dimx       = metadata['dimx'][0][0]
colToCoord = metadata['colToCoord']
coordToCol = metadata['coordToCol']
rois       = metadata['rois']
colToROI   = metadata['colToROI']
###########################################

###########################################
info = starplus['info'][0]
#info: This variable defines the experiment in terms of a sequence of 'trials'. 'info' is a 1x54 struct array, describing the 54 time intervals, or trials. Most of these time intervals correspond to trials during which the subject views a single picture and a single sentence, and presses a button to indicate whether the sentence correctly describes the picture. Other time intervals correspond to rest periods. The relevant fields of info are illustrated in the following example:
#info(18) mint: 894 maxt: 948 cond: 2 firstStimulus: 'P' sentence: ''It is true that the star is below the plus.'' sentenceRel: 'below' sentenceSym1: 'star' sentenceSym2: 'plus' img: sap actionAnswer: 0 actionRT: 3613
#info.mint gives the time of the first image in the interval (the minimum time)
#info.maxt gives the time of the last image in the interval (the maximum time)
#info.cond has possible values 0,1,2,3. Cond=0 indicates the data in this segment should be ignored. Cond=1 indicates the segment is a rest, or fixation interval. Cond=2 indicates the interval is a sentence/picture trial in which the sentence is not negated. Cond=3 indicates the interval is a sentence/picture trial in which the sentence is negated.
#info.firstStimulus: is either 'P' or 'S' indicating whether this trail was obtained during the session is which Pictures were presented before sentences, or during the session in which Sentences were presented before pictures. The first 27 trials have firstStimulus='P', the remained have firstStimulus='S'. Note this value is present even for trials that are rest trials. You can pick out the trials for which sentences and pictures were presented by selecting just the trials trials with info.cond=2 or info.cond=3.
#info.sentence gives the sentence presented during this trial. If none, the value is '' (the empty string). The fields info.sentenceSym1, info.sentenceSym2, and info.sentenceRel describe the two symbols mentioned in the sentence, and the relation between them.
#info.img describes the image presented during this trial. For example, 'sap' means the image contained a 'star above plus'. Each image has two tokens, where one is above the other. The possible tokens are star (s), plus (p), and dollar (d).
#info.actionAnswer: has values -1 or 0. A value of 0 indicates the subject is expected to press the answer button during this trial (either the 'yes' or 'no' button to indicate whether the sentence correctly describes the picture). A value of -1 indicates it is inappropriate for the subject to press the answer button during this trial (i.e., it is a rest, or fixation trial).
#info.actionRT: gives the reaction time of the subject, measured as the time at which they pressed the answer button, minus the time at which the second stimulus was presented. Time is in milliseconds. If the subject did not press the button at all, the value is 0.
###########################################

###########################################
data = starplus['data']
#data: This variable contains the raw observed data. The fMRI data is a sequence of images collected over time, one image each 500 msec. The data structure 'data' is a [54x1] cell array, with one cell per 'trial' in the experiment. Each element in this cell array is an NxV array of observed fMRI activations. The element data{x}(t,v) gives the fMRI observation at voxel v, at time t within trial x. Here t is the within-trial time, ranging from 1 to info(x).len. The full image at time t within trial x is given by data{x}(t,:).
#Note the absolute time for the first image within trial x is given by info(x).mint.
###########################################

def HingeLoss(X, Y, W, lmda):
    #TODO: Compute (regularized) Hinge Loss
    loss = 0
    for i in range(X.shape[0]):
        x = X[i,]
        y = Y[i,]
        y_hat = (x.dot(W))
        loss += max( 0,1-(y * y_hat) )
    loss += lmda * np.sum( np.square (W) )
    return loss

def SgdHinge(X, Y, maxIter, learningRate, lmda):
    W = np.zeros(X.shape[1])

    # TODO: implement stochastic (sub) gradient descent with the hinge loss function

    iter = 0
    p_loss = 0
    c_loss = HingeLoss(X, Y, W, lmda)
    while ( abs(c_loss - p_loss) > 0.0001 and iter < maxIter ):
        print("Iteration %s HINGE-LOSS %s" %(iter, c_loss))
        for i in range(X.shape[0]):
            x = X[i,]
            y = Y[i,]
            dum = 1 - y* W.dot(x)
            part = 0
            if dum >= 0:
                part = (-y* x )
            #W -= learningRate * lmda * W
            W -= learningRate * ( part + (2 * W *lmda) )

        p_loss = c_loss
        c_loss = HingeLoss(X, Y, W, lmda)
        iter += 1

    return W

def LogisticLoss(X, Y, W, lmda):
    #TODO: Compute (regularized) Logistic Loss
    loss = 0
    for i in range(X.shape[0]):
        x = X[i,]
        y = Y[i,]
        y_hat = (x.dot(W))
        # loss += np.log (1+np.exp( -y.dot(y_hat) ) )
        loss += np.log(1 + np.exp(-y*y_hat))
    loss += lmda * np.sum (np.square (W))
    return loss


def SgdLogistic(X, Y, maxIter, learningRate, lmda):
    W = np.zeros(X.shape[1])
    #TODO: implement stochastic gradient descent using the logistic loss function

    iter = 0

    p_loss = 0
    c_loss = LogisticLoss(X, Y, W, lmda)
    while ( abs(p_loss - c_loss) > 0.0001 and iter  < maxIter ):
        print("Iteration %s LOGISTIC LOSS: %s" % (iter, c_loss))
        # print(LogisticLoss(X, Y, W, lmda))
        for i in range(X.shape[0]):
            x = X[i,]
            y = Y[i,]
            dum = np.exp( -1* y *W.dot(x))
            P = dum / (1+dum)
            exp = np.power(np.e, -Y[i]* np.dot(W, np.transpose(X[i,])))
            #W -= learningRate * lmda * W        
            W -= learningRate * ( -(y * P * x)  +2*lmda*W )
        p_loss = c_loss
        c_loss = LogisticLoss(X, Y, W, lmda)
        iter += 1

    return W



def crossValidation(X, Y, SGD, lmda, learningRate, maxIter=100, sample=range(20)):
    #Leave one out cross validation accuracy
    nCorrect   = 0.
    nIncorrect = 0.
    
    for i in sample:
        print ("CROSS VALIDATION %s" % i)
        
        training_indices = [j for j in range(X.shape[0]) if j != i]
        W = SGD(X[training_indices,], Y[training_indices,], maxIter=maxIter, lmda=lmda, learningRate=learningRate)
        print (W)

        ############################################################################################################################
            #### BONUS PART: UNCOMMENT TO OPERATE ####
        # # Average Weight for each ROI
        # MAX_el = max(rois[0][0][2][0])
        # W_collapsed = np.zeros(MAX_el)
        # for j in range(W.shape[0]):
        #     W_collapsed[j % MAX_el] += W[j]
        # W_collapsed /= (W.shape[0] / MAX_el)
        #
        # w_roi_el = namedtuple('W_rois_element', 'Weight Name')
        # w_roi = []
        # for tt in range(rois.shape[1]):
        #     SUM_W = 0
        #     for ll in range(rois[0][tt][2].shape[1]):
        #         SUM_W += abs(W_collapsed[rois[0][tt][2][0][ll] - 1])
        #     w_roi.append(w_roi_el(rois[0][tt][0], SUM_W))
        #
        # print( w_roi)

        ############################################################################################################################

        y_hat = np.sign(X[i,].dot(W))

        if y_hat == Y[i]:
            nCorrect += 1
        else:
            nIncorrect += 1

    return nCorrect / (nCorrect + nIncorrect)

def Accuracy(X, Y, W):
    Y_hat = np.sign(X.dot(W))
    correct = (Y_hat == Y)
    return float(sum(correct)) / len(correct)

def main():
    maxFeatures =  max([data[i][0].flatten().shape[0] for i in range(data.shape[0])])

    #Inputs
    X = np.zeros((ntrials, maxFeatures+1))
    for i in range(data.shape[0]):
        f = data[i][0].flatten()
        X[i,:f.shape[0]] = f
        X[i,f.shape[0]]  = 1     #Bias

    #Outputs (+1 = Picture, -1 = Sentence)
    Y = np.ones(ntrials)
    Y[np.array([info[i]['firstStimulus'][0] != 'P' for i in range(ntrials)])] = -1

    #Randomly permute the data
    np.random.seed(1)      #Seed the random number generator to preserve the dev/test split
    permutation = np.random.permutation(ntrials)
    permutation = np.random.permutation(X.shape[0])
    X = X[permutation,]
    Y = Y[permutation,]

    #Cross validation
    #Development

    # eta = [0.1,0.01,0.001,0.0001]
    # lam = [1, 0.3, 0.1]
    #
    # result =[]
    # for e in eta:
    #     for l in lam:
    #         result.append( (e,l,crossValidation(X, Y, SgdLogistic, maxIter=100, lmda=l, learningRate=e, sample=range(20)))  )
            #print("Accuracy (Logistic Loss):\t%s" % crossValidation(X, Y, SgdLogistic, maxIter=100, lmda=l, learningRate=e, sample=range(20)))

    #print ("Accuracy (Logistic Loss):\t%s" % crossValidation(X, Y, SgdLogistic, maxIter=100, lmda=0.3, learningRate=0.0001, sample=range(20)) )
    #print ("Accuracy (Hinge Loss):\t%s" % crossValidation(X, Y, SgdHinge, maxIter=100, lmda=1, learningRate=0.0001, sample=range(20)) )

    #Test
    #print ("Accuracy (Logistic Loss):\t%s" % crossValidation(X, Y, SgdLogistic, maxIter=10, lmda=0.3, learningRate=0.0001, sample=range(20,X.shape[0])) )
    #print ("Accuracy (Hinge Loss):\t%s" % crossValidation(X, Y, SgdHinge, maxIter=100, lmda=1, learningRate=0.0001, sample=range(20,X.shape[0])) )

# Plotting function
def pl(result, lamda):
    plt.figure
    x = [a for (a,b) in result]
    y = [b for (a,b) in result]
    plt.scatter (x,y)
    plt.title("Lambda="+str(lamda))
    plt.savefig("result_l="+str(lamda)+"_with_Hinge"+".pdf")
    plt.clf()
    plt.cla()
    plt.close('all')



if __name__ == "__main__":
    #main()
    maxFeatures = max([data[i][0].flatten().shape[0] for i in range(data.shape[0])])

    # Inputs
    X = np.zeros((ntrials, maxFeatures + 1))
    for i in range(data.shape[0]):
        f = data[i][0].flatten()
        X[i, :f.shape[0]] = f
        X[i, f.shape[0]] = 1  # Bias

    # Outputs (+1 = Picture, -1 = Sentence)
    Y = np.ones(ntrials)
    Y[np.array([info[i]['firstStimulus'][0] != 'P' for i in range(ntrials)])] = -1

    # Randomly permute the data
    np.random.seed(1)  # Seed the random number generator to preserve the dev/test split
    permutation = np.random.permutation(ntrials)
    permutation = np.random.permutation(X.shape[0])
    X = X[permutation,]
    Y = Y[permutation,]

    # Cross validation
    # Development
##################################################################################################################################################################

   ### Uncomment to generate the Parameter Tunning and corresponding figures ###
    # eta = [0.1, 0.01, 0.001, 0.0001]
    # lam = [1, 0.3, 0.1]
    # result = []
    #
    # plt.figure
    # cycol = cycle('bgrkmyc')
    # for e in eta:
    #     sh = []
    #     for l in lam:
    #         sh.append( crossValidation(X, Y, SgdLogistic, maxIter=100, lmda=l, learningRate=e, sample=range(20)) )
    #     cl = next(cycol)
    #     plt.scatter(lam, sh, s=80+randint(0,5),facecolors='none',edgecolors=cl , label=r"$\eta=$" +str(e))
    #     plt.plot(lam, sh, color=cl, linewidth=2)
    # plt.xlabel(r"$\lambda$")
    # plt.ylabel("Accuracy")
    # #plt.title("Convergence Criterion: Max Iteration=100")
    # plt.title("Hinge")
    # plt.legend()
    # plt.savefig ("Results.pdf")
    # plt.clf()
    # plt.cla()
    # plt.close("all")

##################################################################################################################################################################


    #print ("Accuracy (Logistic Loss):\t%s" % crossValidation(X, Y, SgdLogistic, maxIter=100, lmda=1, learningRate=0.0001, sample=range(20)) )
    #print ("Accuracy (Hinge Loss):\t%s" % crossValidation(X, Y, SgdHinge, maxIter=100, lmda=1, learningRate=0.0001, sample=range(20)) )

    # Test
    print ("Accuracy (Logistic Loss):\t%s" % crossValidation(X, Y, SgdLogistic, maxIter=100, lmda=1, learningRate=0.0001, sample=range(20,X.shape[0])) )
    #print ("Accuracy (Hinge Loss):\t%s" % crossValidation(X, Y, SgdHinge, maxIter=100, lmda=1, learningRate=0.0001, sample=range(20,X.shape[0])) )

print('done')
