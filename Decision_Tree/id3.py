# id3.py
# --------------
# Licensing Information:  You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to The Ohio State University, including a link to http://aritter.github.io/courses/5523_fall18.html
#
# Attribution Information: This assignment was developed at The Ohio State University
# by Alan Ritter (ritter.1492@osu.edu).

from collections import namedtuple
import sys
import math
from Data import *

DtNode = namedtuple("DtNode", "fVal, nPosNeg, gain, left, right")

POS_CLASS = 'e'

def InformationGain(data, f):
    # Initialization
    n_current_nPosNeg= [0,0]
    n_left_child_nPosNeg = [0,0]
    n_right_child_nPosNeg = [0,0]

    #Computing the node nPosNeg
    for d_inst in data:
        if d_inst[f.feature]==f.value:
            if d_inst[0]==POS_CLASS:
                n_current_nPosNeg[0] += 1
                n_left_child_nPosNeg[0] += 1
            else:
                n_current_nPosNeg[1] += 1
                n_left_child_nPosNeg[1] += 1
        else:
            if d_inst[0]==POS_CLASS:
                n_current_nPosNeg[0] += 1
                n_right_child_nPosNeg[0] += 1
            else:
                n_current_nPosNeg[1] += 1
                n_right_child_nPosNeg[1] +=1

 #Computing probabilities and entropies
    p_edible = float(n_current_nPosNeg[0]/ (n_current_nPosNeg[0] + n_current_nPosNeg[1] ) )
    #p_edible = 1 if p_edible == 0 else p_edible
    p_poison = 1-p_edible
    #p_poison = 1 if p_poison == 0 else p_poison
    # print(p_edible)
    # print(p_poison)
    entropy = -p_edible*math.log(p_edible,2)-p_poison*math.log(p_poison,2)

    p_left = float( ( n_left_child_nPosNeg[0] + n_left_child_nPosNeg[1] ) / (n_current_nPosNeg[0]+n_current_nPosNeg[1]) )
    p_right= float( (n_right_child_nPosNeg[0]+n_right_child_nPosNeg[1]) / (n_current_nPosNeg[0]+n_current_nPosNeg[1]) )


    p_edible_left_child = 0 if (n_left_child_nPosNeg[0]+n_left_child_nPosNeg[1] ) == 0 else float( n_left_child_nPosNeg[0]/ (n_left_child_nPosNeg[0]+n_left_child_nPosNeg[1] ) )
    p_edible_left_child = 1 if p_edible_left_child ==0 else p_edible_left_child
    p_poison_left_child = 1-p_edible_left_child
    p_poison_left_child = 1 if p_poison_left_child ==0 else p_poison_left_child

    p_edible_right_child = 0 if (n_right_child_nPosNeg[0]+n_right_child_nPosNeg[1]) == 0 else float( n_right_child_nPosNeg[0]/(n_right_child_nPosNeg[0]+n_right_child_nPosNeg[1]) )
    p_edible_right_child = 1 if p_edible_right_child == 0 else p_edible_right_child
    p_poison_right_child = 1-p_edible_right_child
    p_poison_right_child = 1 if p_poison_right_child == 0 else p_poison_right_child

    print(p_edible_left_child)
    print(p_edible_right_child)
    entropy_left = -p_edible_left_child * math.log(p_edible_left_child,2) - p_poison_left_child * math.log(p_poison_left_child,2)
    entropy_right = -p_edible_right_child * math.log(p_edible_right_child,2) - p_poison_right_child * math.log(p_poison_right_child,2)

    information_gain = entropy - (p_left * entropy_left + p_right * entropy_right)

    return information_gain

def Classify(tree, instance):
    if tree.left == None and tree.right == None:
        return tree.nPosNeg[0] > tree.nPosNeg[1]
    elif instance[tree.fVal.feature] == tree.fVal.value:
        return Classify(tree.left, instance)
    else:
        return Classify(tree.right, instance)

def Accuracy(tree, data):
    nCorrect = 0
    for d in data:
        if Classify(tree, d) == (d[0] == POS_CLASS):
            nCorrect += 1
    return float(nCorrect) / len(data)

def PrintTree(node, prefix=''):
    print("%s>%s\t%s\t%s" % (prefix, node.fVal, node.nPosNeg, node.gain))
    if node.left != None:
        PrintTree(node.left, prefix + '-')
    if node.right != None:
        PrintTree(node.right, prefix + '-')        
        
def ID3(data, features, MIN_GAIN=0.1):
    #TODO: implement decision tree learning
    best_feature =  DtNode(None, (0,0), 0, None, None)

    #Evaluating number of edibles
    num_edibles = 0
    for inst in data:
        if inst[0] == POS_CLASS:
            num_edibles += 1
    best_feature = best_feature._replace(nPosNeg=[num_edibles,len(data)-num_edibles])

    # Error Finding:
    if num_edibles == 0 or num_edibles == len(data) or len(features) == 0 :
        return best_feature

    # Going over the features and see what is the best feature
    for feat_val in features:
        information_gain = InformationGain(data,feat_val)
        if information_gain > best_feature.gain:
            best_feature = best_feature._replace(gain=information_gain)
            best_feature = best_feature._replace(fVal=feat_val)

    if best_feature.gain < MIN_GAIN :
        return best_feature

    right = []
    left = []

    for instance in data:
        if instance[best_feature.fVal.feature] == best_feature.fVal.value:
            left.append(instance)
        else:
            right.append(instance)

    if len(left) == 0:
        best_feature = best_feature._replace(left=DtNode(None, (num_edibles, len(data)-num_edibles),0,None,None ))
    else:
        best_feature = best_feature._replace(left=ID3(left,features,MIN_GAIN=0.1))

    if len(right) == 0:
        best_feature = best_feature._replace(right = DtNode(None, (num_edibles,len(data)-num_edibles),0,None, None))
    else:
        best_feature = best_feature._replace(right=ID3(right,features,MIN_GAIN=0.1))

    return best_feature


    # return DtNode(FeatureVal(1,'x'), (100,0), 0, None, None)

if __name__ == "__main__":
    # train = MushroomData(sys.argv[1])
    # dev = MushroomData(sys.argv[2])


    train = MushroomData("train")
    dev = MushroomData("test")

    # dTree = ID3(train.data, train.features, MIN_GAIN=float(sys.argv[3]))
    dTree = ID3(train.data, train.features, MIN_GAIN=float(0))
    
    PrintTree(dTree)

    print (Accuracy(dTree, dev.data) )

    print("done")
