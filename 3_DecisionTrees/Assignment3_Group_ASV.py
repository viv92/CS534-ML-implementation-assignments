import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import random


class DecisionTree:
  def __init__(self, fin , th, lf, pp, depth):
    self.fIndex = fin
    self.threshold = th
    self.leafFlag = lf
    self.probplus = pp
    self.depth = depth

  def attachLeftNode(self, LN):
    self.leftNode = LN

  def attachRightNode(self, RN):
    self.rightNode = RN

DEPTH = 0
#========= Main Function
def main():
    X_training, Y_training, X_validation, Y_validation = preprocessing ('pa3_train_reduced.csv','pa3_valid_reduced.csv')
    global DEPTH


    #reducing number of features for debugging

#------------- PART 1----------------
    Part1(X_training, Y_training, X_validation, Y_validation)

#------------- PART 2----------------
    Part2_main(X_training,Y_training,X_validation, Y_validation)
#------------------------------------
    Part3(X_training,Y_training,X_validation, Y_validation)





def Part1(X_training, Y_training, X_validation, Y_validation):
    prob_plus_initial = calc_prob(Y_training)


    #loop to get accuracies vs depth
    acc_tr = []
    acc_val = []
    d = []
    d_max = 20
    for md in range(0, d_max+1):

        root = dt_recursion(X_training, Y_training, prob_plus_initial, 0, md)
        #calculate accuracy
        accuracy_train = get_accuracy(X_training, Y_training, root, md)
        accuracy_val = get_accuracy(X_validation, Y_validation, root, md)
        print("DEPTH: ", DEPTH)
#        print("accuracy_tr: ", accuracy_train)
#        print("accuracy_val: ", accuracy_val)
        if len(d) > 0 and DEPTH == d[-1]:
            break
        d.append(DEPTH)
        acc_tr.append(accuracy_train)
        acc_val.append(accuracy_val)

    #plot accuracies vs depth
    #d = np.arange(0, len(acc_tr))
    plt.figure()
    plt.plot(d, acc_tr, label = "Training Accuracy",color='red', linestyle='solid', linewidth = 1.5 )
    plt.plot(d, acc_val, label = "Validation Accuracy",color='blue', linestyle='solid', linewidth = 1.5 )
    plt.xlabel('Depth of Tree', fontsize=12)
    plt.ylabel('Accuracies \n max_validation_accuracy=' + str(np.max(acc_val)) + " at depth=" + str(np.argmax(acc_val)) + '\n max_training_accuracy=' + str(np.max(acc_tr)) + " at depth=" + str(np.argmax(acc_tr))  , fontsize=11)
    plt.title('Accuracies vs Depth of Decision Tree', fontsize=11)
    plt.legend()
    plt.grid()
    # changingY-axis range
    x1,x2,y1,y2 = plt.axis()
    plt.savefig("Part1_Accuracies_vs_Depth.png", dpi = 300,bbox_inches="tight")
    plt.show()


def Part2_main(X_training,Y_training,X_validation, Y_validation):
    print('=========================')
    print(">>Part 2: working...")
    print('=========================')
    d=9
    m=10
    print("m=",m)
    n=[1,3,5,10,15,20,25,30]
    validation_accuracies = np.zeros(len(n))
    training_accuracies = np.zeros(len(n))
    for i in range(0,len(n)):
        print("No.of.Trees:", n[i])
        [training_accuracies[i], validation_accuracies[i]]  = random_forest(X_training,Y_training,X_validation, Y_validation, n[i],m,d)
    plot_part2(training_accuracies, validation_accuracies, n, d, m)

    m=20
    print("m=",m)
    validation_accuracies = np.zeros(len(n))
    training_accuracies = np.zeros(len(n))
    for i in range(0,len(n)):
        print("No.of.Trees:", n[i])
        [training_accuracies[i], validation_accuracies[i]]  = random_forest(X_training,Y_training,X_validation, Y_validation, n[i],m,d)
    plot_part2(training_accuracies, validation_accuracies, n, d, m)

    m=50
    print("m=",m)
    validation_accuracies = np.zeros(len(n))
    training_accuracies = np.zeros(len(n))
    for i in range(0,len(n)):
          print("No.of.Trees:", n[i])
          [training_accuracies[i], validation_accuracies[i]]  = random_forest(X_training,Y_training,X_validation, Y_validation, n[i],m,d)
    plot_part2(training_accuracies, validation_accuracies, n, d, m)




def calc_accuracy(x,y,node):

    #reached leaf
    if node.leafFlag == True:
        if (node.probplus > 0.5 and y == 1) or (node.probplus < 0.5 and y == -1):
            return True
        else:
            return False
    #recursion case
    if x[node.fIndex] >= node.threshold:
        return calc_accuracy(x,y,node.rightNode)
    else:
        return calc_accuracy(x,y,node.leftNode)

def get_accuracy(X, Y, root, max_depth):
    correct_pred = 0
    for i in range(X.shape[0]):
        if calc_accuracy(X[i], Y[i], root):
            correct_pred += 1
    accuracy = (correct_pred*100)/X.shape[0]
    return accuracy

def calc_prob(Y):
    return ( ((np.sum(Y)/len(Y)) + 1) / 2 )

def calc_accuracy_part2(x,node):
    #reached leaf
    if node.leafFlag == True:
        if (node.probplus > 0.5):
            return 1
        else:
            return -1
    #recursion case
    if x[node.fIndex] >= node.threshold:
        return calc_accuracy_part2(x,node.rightNode)
    else:
        return calc_accuracy_part2(x,node.leftNode)

def get_accuracy_part2(X, Y,  random_forest_nodes, max_depth):
    correct_pred = 0
    for i in range(X.shape[0]):
        majorityVote = 0
        for j in range(len(random_forest_nodes)):
            majorityVote += calc_accuracy_part2(X[i], random_forest_nodes[j])
        if (np.sign(majorityVote) == Y[i]):
            correct_pred += 1
    accuracy = (correct_pred*100)/X.shape[0]
    return accuracy




def dt_recursion(X, Y, prob_plus, depth, max_depth):
    global DEPTH
    if DEPTH < depth:
        DEPTH += 1
    #print("building new node")

    #base case - leaf node

    if prob_plus == 1 or prob_plus == 0 or depth == max_depth:
#        print("----------making leaf node at depth = ", depth)
        node = DecisionTree(-2, -2, True, prob_plus, depth)
        return node


    #get best feature and best threshold
#    print("finding best feature for new split")
    best_feature_index, best_threshold, left_prob_plus, right_prob_plus= findBestFeature(X, Y)
#    print("found best feature: ", best_feature_index)

    #create empty containers for split data
    X_left = []
    Y_left = []
    X_right = []
    Y_right = []

    #create split

    for i in range(0, X.shape[0]):
        if X[i][best_feature_index] < best_threshold:
            if len(X_left) == 0:
                X_left = X[i]
                Y_left = Y[i]
            else:
                X_left = np.vstack((X_left, X[i]))
                Y_left = np.vstack((Y_left, Y[i]))
        else:
            if len(X_right) == 0:
                X_right = X[i]
                Y_right = Y[i]
            else:
                X_right = np.vstack((X_right, X[i]))
                Y_right = np.vstack((Y_right, Y[i]))
    #vectorize Y
    Y_left = np.reshape(Y_left, (-1))
    Y_right = np.reshape(Y_right, (-1))

    #create local node
#    print("----------making mid node at depth = ", depth)
    node = DecisionTree(best_feature_index, best_threshold, False, prob_plus, depth)

    if len(Y_left) == 0:
        print("threshold at boundary")
    else:
        node.attachLeftNode(dt_recursion(X_left, Y_left, left_prob_plus, depth+1, max_depth))

    if len(Y_right) == 0:
        print("threshold at boundary")
    else:
        node.attachRightNode(dt_recursion(X_right, Y_right, right_prob_plus, depth+1, max_depth))

    return node


def findBestFeature(X, Y):
    final_thresh = 0
    final_left_pp = 0
    final_right_pp = 0
    feature_index = -1
    maxB = -1 * float('inf')

    #print("X.shape", X.shape[0])
    #print("**********X.shape: ", X.shape)
    if len(X.shape) == 1:
        X = np.reshape(X, (-1, 1))
    #print("**********X.shape: ", X.shape)
    for i in range(0, X.shape[1]):
        f = np.zeros(X.shape[0]).astype('float')
        #print("f.shape:", f.shape)
        np.copyto(f, X[:,i])
        #print("********f.shape: ", f.shape)
        #print("********Y.shape: ", Y.shape)
        #print("finding best threshold for feature ", i)
        t, b, left_pp, right_pp = findBestThreshold(f, Y)
        #print("found best threshold")

        if maxB < b:
            maxB = b
            final_thresh = t
            final_left_pp = left_pp
            final_right_pp = right_pp
            feature_index = i

    return(feature_index, final_thresh, final_left_pp, final_right_pp)


def findBestThreshold(f, Y):
    maxB = -1 * float('inf')
    final_thresh = 0
    final_left_pp = 0
    final_right_pp = 0

    sorted_f, sorted_Y = zip(*sorted(zip(f, Y)))

    hero = sorted_Y[0]
    for i in range(1, len(sorted_Y)):
        if sorted_Y[i] != hero:
            t = sorted_f[i]
            b, left_pp, right_pp = calculateBenefit(sorted_Y, i)
            if maxB < b:
                maxB = b
                final_thresh = t
                final_left_pp = left_pp
                final_right_pp = right_pp
            hero = sorted_Y[i]

    return(final_thresh, maxB, final_left_pp, final_right_pp)


def calculateBenefit(y, i):
  left = i
  right = len(y) - i

  pl = left/len(y)
  pr = right/len(y)

  clpos = (0.5*(sum(y[0:left])/(left) + 1))
  crpos = (0.5*(sum(y[left:len(y)])/(len(y)-left) + 1))

  benefit = gini(y, 0, len(y)) - (pl*gini(y, 0, left)) - (pr*gini(y, left, len(y)))

  return benefit, clpos, crpos



def gini(y, initial, final):

  pos = (0.5*(sum(y[initial:final])/(final-initial) + 1))
  neg = 1 - pos

  gini = 1 - (pos/(pos+neg))**2 - (neg/(pos+neg))**2

  return gini


def random_forest(X,Y,X_validation, Y_validation, n,m,d):

    random_forest_nodes = np.zeros((n,), dtype=DecisionTree)

    for i in range(0, n):
        #Z_bagging = choices(Z, k=Z.shape[0])

        #Selecting X.shape[o] examples with replacements. Some examples can be repeated and some may not be included in some training sets
        A = np.random.randint(X.shape[0], size=X.shape[0])

        X_bagging = X[A]
        Y_bagging = Y[A]

        prob_plus_initial = calc_prob(Y_bagging)
        random_forest_nodes[i]  = dt_recursion_part2(X_bagging, Y_bagging, prob_plus_initial, 0, np.inf,m)

    accuracy_train = get_accuracy_part2(X, Y, random_forest_nodes, 20)
    accuracy_val = get_accuracy_part2(X_validation, Y_validation, random_forest_nodes, 20)
#    print("accuracy_tr: ", accuracy_train)
#    print("accuracy_val: ", accuracy_val)
#    print("DEPTH: ", DEPTH)
    return (accuracy_train, accuracy_val)

def plot_part2(trainAccuracy, validationAccuracy, number_trees, d,m):
    # print('Inside PlotOnlinePerceptron method')
    # '''print('accuracy: ',accuracy)
    # print('iterations: ',iterations)'''

    plt.plot(number_trees,trainAccuracy, label='Training Accuracy', color='red', linewidth=1.5,marker=".")
    plt.plot(number_trees,validationAccuracy, label='Validation Accuracy', color='blue', linewidth=1.5,marker=".")
    plt.title('Random Forest: Training and Validation Accuracy vs No. of trees ' )

    plt.legend(loc='best')
    plt.xlabel(  ("No. of Trees \nMax Depth(d) ="+str(d)+', No. of features selected(m)='+ str(m) ))
    plt.ylabel('Accuracy \nMax Training: '+str(np.max(trainAccuracy))+' \n Max Validation: '+str(np.max(validationAccuracy)))
    plt.grid()
    #print('Dimension of X: {}, Dimesion of Y: '.format(len(accuracy), len(iterations)))
    plt.savefig("Part2_AccuracyVsno_of_trees_depth_m_"+str(m)+".png", dpi = 300,bbox_inches="tight")
    plt.show()


def dt_recursion_part2(X, Y, prob_plus, depth, max_depth, m_features):

    global DEPTH
    if DEPTH < depth:
        DEPTH += 1

    if prob_plus == 1 or prob_plus == 0 or depth == max_depth:

        node = DecisionTree(-2, -2, True, prob_plus, depth)
        return node


    #get best feature and best threshold

    best_feature_index, best_threshold, left_prob_plus, right_prob_plus= findBestFeature_part2(X, Y, m_features)


    #create empty containers for split data
    X_left = []
    Y_left = []
    X_right = []
    Y_right = []



    for i in range(0, X.shape[0]):

        if X[i][best_feature_index] < best_threshold:
            if len(X_left) == 0:
                X_left = X[i]
                Y_left = Y[i]
            else:
                X_left = np.vstack((X_left, X[i]))
                Y_left = np.vstack((Y_left, Y[i]))
        else:
            if len(X_right) == 0:
                X_right = X[i]
                Y_right = Y[i]
            else:
                X_right = np.vstack((X_right, X[i]))
                Y_right = np.vstack((Y_right, Y[i]))
    #vectorize Y
    Y_left = np.reshape(Y_left, (-1))
    Y_right = np.reshape(Y_right, (-1))
    #create local node
    node = DecisionTree(best_feature_index, best_threshold, False, prob_plus, depth)

    if len(Y_left) == 0:
        print("threshold at boundary")
    else:
        node.attachLeftNode(dt_recursion_part2(X_left, Y_left, left_prob_plus, depth+1, max_depth,m_features))

    if len(Y_right) == 0:
        print("threshold at boundary")
    else:
        node.attachRightNode(dt_recursion_part2(X_right, Y_right, right_prob_plus, depth+1, max_depth,m_features))

    return node


def findBestFeature_part2(X, Y, m_features):
    final_thresh = 0
    final_left_pp = 0
    final_right_pp = 0
    feature_index = -1
    maxB = -1 * float('inf')

    if len(X.shape) == 1:
        X = np.reshape(X, (-1, 1))
    # selecting m features when spliting a node. Features will be selected without replacements
    selected_features = random.sample(range(X.shape[1]), m_features)
    for i in selected_features:

        f = np.zeros(X.shape[0]).astype('float')
        #print("f.shape:", f.shape)
        np.copyto(f, X[:,i])

        t, b, left_pp, right_pp = findBestThreshold(f, Y)

        if maxB < b:
            maxB = b
            final_thresh = t
            final_left_pp = left_pp
            final_right_pp = right_pp
            feature_index = i

    return(feature_index, final_thresh, final_left_pp, final_right_pp)


def Part3_get_accuracy(X, Y, all_trees, all_alphas):
    correct_pred = 0
    for i in range(X.shape[0]):
        final_pred = 0
        for j in range(len(all_trees)):
            if calc_accuracy(X[i], Y[i], all_trees[j]):
                pred_lb = Y[i]
            else:
                pred_lb = -1 * Y[i]
            final_pred += all_alphas[j] * pred_lb
        if final_pred >= 0:
            final_pred = 1
        else:
            final_pred = -1
        if final_pred == Y[i]:
            correct_pred += 1
    accuracy = (correct_pred*100)/X.shape[0]
    return accuracy


def calc_epsilon(X, Y, root, max_depth):
    eps = 0
    for i in range(X.shape[0]):
        if not calc_accuracy(X[i], Y[i][0], root):
            eps += Y[i][1]
    return eps

def update_distribution(X, Y, root, alpha):
    for i in range(X.shape[0]):
        if calc_accuracy(X[i], Y[i][0], root):
            Y[i][1] *= math.exp(-alpha)
        else:
            Y[i][1] *= math.exp(alpha)
    s = sum(n for _, n in Y)
    for i in range(len(Y)):
        Y[i][1] /= s
    return Y


def Part3(X_training, Y_training, X_validation, Y_validation):


    #initial weight distribution (uniform)
    w = np.ones(X_training.shape[0])
    w /= X_training.shape[0]



    #weightify data (labels)
    Yw_training = [[Y_training[i], w[i]] for i in range(0, len(Y_training))]

    #calculate initial probability
    prob_plus_initial = calc_positive_prob(Yw_training)

    # L-loop
    L_list = [1, 5, 10, 20]
    acc_tr = []
    acc_val = []
    DEPTH = 9
    for L in L_list:
        all_trees = []
        all_alphas = []
        for l in range(L):
            #train tree
            root = Part3_dt_recursion(X_training, Yw_training, prob_plus_initial, 0, DEPTH)

            all_trees.append(root)
            #calculate epsilon
            epsilon = calc_epsilon(X_training, Yw_training, root, DEPTH)

            #calculate alpha
            alpha = 0.5 * math.log((1-epsilon)/epsilon)
            all_alphas.append(alpha)
            #update weights
            Yw_training = update_distribution(X_training, Yw_training, root, alpha)


        #calculate accuracy
        accuracy_train = Part3_get_accuracy(X_training, Y_training, all_trees, all_alphas)
        accuracy_val = Part3_get_accuracy(X_validation, Y_validation, all_trees, all_alphas)
        acc_tr.append(accuracy_train)
        acc_val.append(accuracy_val)
        print("L: ", L, " accuracy_train: ", accuracy_train, "accuracy_val: ", accuracy_val)

    #plot graph
    plt.figure()
    plt.plot(L_list, acc_tr, label = "Training Accuracy",color='red', linestyle='solid', linewidth = 1.5 )
    plt.plot(L_list, acc_val, label = "Validation Accuracy",color='blue', linestyle='solid', linewidth = 1.5 )
    plt.xlabel('L', fontsize=12)
    plt.ylabel('Accuracies \n max_validation_accuracy=' + str(np.max(acc_val)) + " at L=" + str(L_list[np.argmax(acc_val)]) + '\n max_training_accuracy=' + str(np.max(acc_tr)) + " at L=" + str(L_list[np.argmax(acc_tr)])  , fontsize=11)
    plt.title('Accuracies vs L values', fontsize=11)
    plt.legend(loc=7)
    plt.grid()
    # changingY-axis range
    x1,x2,y1,y2 = plt.axis()
    plt.savefig("Part3_Accuracies_vs_L.png", dpi = 300,bbox_inches="tight")
    plt.show()



def Part3_dt_recursion(X, Y, prob_plus, depth, max_depth):

    #base case - leaf node
    if prob_plus == 1 or prob_plus == 0 or depth == max_depth:
        node = DecisionTree(-2, -2, True, prob_plus, depth)
        return node


    #get best feature and best threshold
    best_feature_index, best_threshold, left_prob_plus, right_prob_plus= Part3_findBestFeature(X, Y)

    #create empty containers for split data
    X_left = []
    Y_left = []
    X_right = []
    Y_right = []

    for i in range(0, X.shape[0]):
        if X[i][best_feature_index] < best_threshold:
            if len(X_left) == 0:
                X_left = X[i]
                Y_left = Y[i]
            else:
                X_left = np.vstack((X_left, X[i]))
                Y_left = np.vstack((Y_left, Y[i]))
        else:
            if len(X_right) == 0:
                X_right = X[i]
                Y_right = Y[i]
            else:
                X_right = np.vstack((X_right, X[i]))
                Y_right = np.vstack((Y_right, Y[i]))

    #create local node
    node = DecisionTree(best_feature_index, best_threshold, False, prob_plus, depth)

    if len(Y_left) == 0:
        print("threshold at boundary")
    else:
        node.attachLeftNode(Part3_dt_recursion(X_left, Y_left, left_prob_plus, depth+1, max_depth))

    if len(Y_right) == 0:
        print("threshold at boundary")
    else:
        node.attachRightNode(Part3_dt_recursion(X_right, Y_right, right_prob_plus, depth+1, max_depth))

    return node


def Part3_findBestFeature(X, Y):
    final_thresh = 0
    final_left_pp = 0
    final_right_pp = 0
    feature_index = -1
    maxB = -1 * float('inf')

    if len(X.shape) == 1:
        X = np.reshape(X, (-1, 1))
    for i in range(0, X.shape[1]):
        f = np.zeros(X.shape[0]).astype('float')
        np.copyto(f, X[:,i])
        t, b, left_pp, right_pp = Part3_findBestThreshold(f, Y)
        if maxB < b:
            maxB = b
            final_thresh = t
            final_left_pp = left_pp
            final_right_pp = right_pp
            feature_index = i

    return(feature_index, final_thresh, final_left_pp, final_right_pp)


def Part3_findBestThreshold(f, Y):
    maxB = -1 * float('inf')
    final_thresh = 0
    final_left_pp = 0
    final_right_pp = 0

    y0 = [n for n,_ in Y]
    y1 = [n for _,n in Y]
    sorted_f, sorted_y0 = zip(*sorted(zip(f, y0)))
    sorted_f, sorted_y1 = zip(*sorted(zip(f, y1)))
    sorted_Y = [[sorted_y0[i], sorted_y1[i]] for i in range(len(sorted_y0))]


    hero = sorted_Y[0][0]
    for i in range(1, len(sorted_Y)):
        if sorted_Y[i][0] != hero:
            t = sorted_f[i]
            b, left_pp, right_pp = Part3_calculateBenefit(sorted_Y, i)
            if maxB < b:
                maxB = b
                final_thresh = t
                final_left_pp = left_pp
                final_right_pp = right_pp
            hero = sorted_Y[i][0]

    return(final_thresh, maxB, final_left_pp, final_right_pp)

def calc_positive_prob(Yw):
    numerator = 0
    denominator = 0
    for i in range(len(Yw)):
        if Yw[i][0] == 1:
            numerator += Yw[i][1]
        denominator += Yw[i][1]
    return numerator/float(denominator)

def weighted_count(Yw):
    return sum(n for _, n in Yw)

def Part3_calculateBenefit(y, i):
  left = i
  #right = len(y) - i
  denom = weighted_count(y)
  pl = weighted_count(y[0:left]) / denom
  pr = weighted_count(y[left:len(y)]) / denom
  clpos = calc_positive_prob(y[0:left])
  crpos = calc_positive_prob(y[left:len(y)])
  cpos = calc_positive_prob(y)
  benefit = Part3_gini2(cpos) - (pl*Part3_gini2(clpos)) - (pr*Part3_gini2(crpos))
  return benefit, clpos, crpos


def Part3_gini2(pos_prob):
    neg_prob = 1 - pos_prob
    gini = 1 - (pos_prob**2) - (neg_prob**2)
    return gini



def preprocessing(train_csv, dev_csv):
    print('=========================')
    print('Preprocessing Running')
    print('=========================')

    #load data from train csv
    data =  np.genfromtxt(train_csv, delimiter=',', dtype=float)
    y = np.zeros_like(data[:,0])
    np.copyto(y, data[:,0])
     #convert to float
    data = np.asarray(data, dtype=float)
    #print("data.shape1: ", data.shape)
    y = np.asarray(y, dtype=float)
    #delete labels from data
    data = np.delete(data, 0, 1)
    #print("data.shape2: ", data.shape)

    #load data from validate csv
    data_valid =  np.genfromtxt(dev_csv, delimiter=',', dtype=float)
    #get targets (price)
    y_valid = np.zeros_like(data_valid[:,0])
    np.copyto(y_valid, data_valid[:,0])
    #convert to float
    data_valid = np.asarray(data_valid, dtype=float)
    y_valid = np.asarray(y_valid, dtype=float)
    #delete labels from data_valid
    data_valid = np.delete(data_valid, 0, 1)


    #assigning labels +1 to 3 and -1 to label 5
    for i in range(0, len(y)):
        if (y[i]==3):
            y[i]=1
        else:
            y[i]=-1


    for i in range(0, len(y_valid)):
        if (y_valid[i]==3):
            y_valid[i]=1
        else:
            y_valid[i]=-1

#    np.savetxt("y.csv", y, delimiter=",")
    print("preproc done")

    return [data, y, data_valid, y_valid]


if __name__ == '__main__':
    main()
