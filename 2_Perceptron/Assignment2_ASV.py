import numpy as np
import math
import matplotlib.pyplot as plt
from io import StringIO
import csv


def main():

    #preprocessing
    X_training,Y_training,X_validation, Y_validation,X_testing  = preprocessing ('pa2_train.csv','pa2_valid.csv','pa2_test_no_label.csv')

    # PART 1 OF THE ASSIGNMENT
    Part1(X_training,Y_training,X_validation, Y_validation,X_testing)

    # PART 2 OF THE ASSIGNMENT
    Part2(X_training,Y_training,X_validation, Y_validation,X_testing, 15 )

    # PART 3 OF THE ASSIGNMENT
    Part3_caller(X_training,Y_training,X_validation,Y_validation,X_testing)


def onlinePerceptron(XTraining, YTraining, XValidation, YValidation, xTest, iterations):
    trainAccuracy = []
    validationAccuracy = []

    actualTrainingPrediction = []
    actualValidationPrediction = []

    YTestPrediction = []

    #w = np.zeros_like(X.shape[1])
    w = np.ones(XTraining.shape[1]) * 0

    #print('w: ',w.shape)
    #print('XTraining: '+str(XTraining.shape)+' YTraining: '+str(YTraining.shape))
    #print('XValidation: '+str(XValidation.shape)+' YValidation: '+str(YValidation.shape))

    for i in range(0,iterations):

        for t in range(0,XTraining.shape[0]):
            ut = np.sign(np.dot(w,XTraining[t, :]))

            if YTraining[t]*ut <= 0:
                w = w + YTraining[t]*XTraining[t, :]

        YTrainPrediction = np.sign(np.matmul(XTraining, w))
        actualTrainingPrediction = (sum(np.abs(YTrainPrediction+YTraining))/2)

        YValidationPrediction = np.sign(np.matmul(XValidation, w))
        actualValidationPrediction = (sum(np.abs(YValidationPrediction+YValidation))/2)

        if i == 13:
            YTestPrediction = np.sign(np.matmul(xTest, w))
            np.savetxt('oplabel.csv', YTestPrediction)

        trainAccuracy.append((100*actualTrainingPrediction)/YTraining.shape[0])
        validationAccuracy.append((100*actualValidationPrediction)/YValidation.shape[0])

#    np.savetxt('oplabel.csv', YTestPrediction)

    return [trainAccuracy, validationAccuracy, YTrainPrediction, YValidationPrediction, YTestPrediction]

def plotOnlinePerceptron(trainAccuracy, validationAccuracy, iterations):
    # print('Inside PlotOnlinePerceptron method')
    # '''print('accuracy: ',accuracy)
    # print('iterations: ',iterations)'''

    plt.plot(trainAccuracy, label='Training Accuracy', color='red', linewidth=1.5)
    plt.plot(validationAccuracy, label='Validation Accuracy', color='blue', linewidth=1.5)
    plt.title('Online Perceptron: Training and Validation Accuracy vs Iterations '+str(iterations))

    plt.legend(loc='best')
    plt.xlabel('Iterations: '+str(iterations))
    plt.ylabel('Accuracy \nMax Training: '+str(np.max(trainAccuracy))+' \n Max Validation: '+str(np.max(validationAccuracy)))
    plt.grid()
    #print('Dimension of X: {}, Dimesion of Y: '.format(len(accuracy), len(iterations)))
    plt.savefig("Part1_AccuracyVs" + str(iterations) + "Iterations.png", dpi = 300,bbox_inches="tight")
    plt.show()


def Part1(xTrain, yTrain, xValidation, yValidation, xTest ):

    print('=========================')
    print('Part 1 Running')
    print('=========================')

    #Getting accuracies for training data
    iterations = 15

    #Getting accuracies for validation data

    [trainAccuracy, validationAccuracy, YTrainPrediction, YValidationPrediction, YTestPrediction] = onlinePerceptron(xTrain, yTrain, xValidation, yValidation, xTest, iterations)
    plotOnlinePerceptron(trainAccuracy, validationAccuracy, iterations)

    print('=========================')
    print('Part 1 Predictions  - oplabel.csv Generated')
    print('=========================')

    # print('trainAccuracy for iterations {}: {}'.format(iterations, trainAccuracy))
    # print('validationAccuracy for iterations {}: {}'.format(iterations, validationAccuracy))

    #np.savetxt('Part1Accuracy.csv', [trainAccuracy, validationAccuracy], delimiter=',')


def Part2(X,Y,X_validation, Y_validation,X_testing, MAX_ITERATIONS):
    #Part 2 goes here
    print('=========================')
    print('Part 2 Running')
    print('=========================')

    accuracy_training = []
    accuracy_validation = []
    w = np.ones(X.shape[1]) * 0
    w_average = np.ones(X.shape[1]) * 0
    c = 0
    s = 0
    for current_iteration in range(1,MAX_ITERATIONS+1):
        for index_sample in range(0,X.shape[0] ):
            ut = np.sign(np.dot(w,X[index_sample,:]) )
            if(Y[index_sample]*ut <= 0):
                if (s+c > 0):
                    w_average = (s*w_average + c*w)/(s+c)
                s = s+c
                w = w + Y[index_sample]*X[index_sample,:]
            else:
                c = c+1
        Y_train_predicted = np.sign( np.matmul(X, w_average)  )
        correct_predictions_training = sum(np.abs(Y_train_predicted + Y))/2
        accuracy_training.append(100*(correct_predictions_training)/Y.shape[0])

        Y_validation_predicted = np.sign( np.matmul(X_validation, w_average)  )
        correct_predictions_validation = sum(np.abs(Y_validation_predicted + Y_validation))/2
        accuracy_validation.append(100*(correct_predictions_validation)/Y_validation.shape[0])

    if (c>0):
        w_average = (s*w_average + c*w)/(s+c)

    plot_Part2(accuracy_training, accuracy_validation, MAX_ITERATIONS)


def plot_Part2(sse_training, sse_validation, max_iteration ):
        #plt.figure(num=None, figsize=(18, 9), dpi=300, facecolor='w', edgecolor='k')
        plt.plot(sse_training, label = "Training Accuracy",color='blue', linestyle='solid', linewidth = 1.5)
        plt.plot(sse_validation, label = "Validation Accuracy",color='red', linestyle='solid', linewidth = 1.5 )

        plt.xlabel('Iterations (total = ' + str(max_iteration) + ' )', fontsize=12)
        plt.ylabel('Accuracy \n max_train = ' + str(np.amax(sse_training)) + ',  \n max_validation = ' + str(np.amax(sse_validation))  , fontsize=11)

        plt.title('Average Perceptron: Accuracy vs Iterations', fontsize=11)
        plt.legend(loc=7)
        plt.grid()
        # changingY-axis range
        x1,x2,y1,y2 = plt.axis()

        plt.savefig("Part2_Accuracy_vs_Iterations.png", dpi = 300,bbox_inches="tight")
        plt.show()


#kernel function
def kp(x,y,p):
    value = math.pow( (1 + np.dot(x,y)) , p)
    return value

def Part3_caller(X_training,Y_training,X_validation, Y_validation,X_testing):
    #container for best validation accuracies, alphas and polynomial degrees
    p = [1,2,3,7,15]
    #p = [1,3]
    best_validation_accuracies = np.zeros(len(p))
    alphas = np.zeros((len(p),X_training.shape[0]))
    #max_iters
    max_iters = 25
    #Part 3 goes here
    print('=========================')
    print('Part 3 Running')
    print('=========================')
    #start
    for i in range(0,len(p)):
        print ('\nStarting for p='+str(p[i]))
        best_validation_accuracies[i], alphas[i] = Part3(X_training,Y_training,X_validation, Y_validation,X_testing, p[i], max_iters)

    #plot best_validation_accuracies vs polynomial degrees
    plt.figure()
    plt.plot(p, best_validation_accuracies, label = "Validation Accuracy",color='red', linestyle='solid', linewidth = 1.5 )
    plt.xlabel('Polynomial Degree', fontsize=12)
    plt.ylabel('Validation Accuracy \n max = ' + str(np.max(best_validation_accuracies))  , fontsize=11)
    plt.title('Polynomial Kernel Perceptron: Validation Accuracy vs Polynomial Degree', fontsize=11)
    plt.legend(loc=7)
    plt.grid()
    # changingY-axis range
    x1,x2,y1,y2 = plt.axis()
    plt.savefig("Part3_Validation_Accuracy_vs_Polynomial_Degree.png", dpi = 300,bbox_inches="tight")
    plt.show()

    #use best alpha to predict for test data
    #print('best_validation_accuracies:')
    #print(best_validation_accuracies)
    best_index = np.argmax(best_validation_accuracies)
    #print('best_index:'+str(best_index))
    best_a = alphas[best_index]
    #print('best_a:'+str(best_a))
    best_p = p[best_index]
    #print('best_p:'+str(best_p))
    y_pred = np.zeros(X_testing.shape[0]) #to store predictions
    #predict on test data
    for index_sample_test in range(0,X_testing.shape[0]):
        u_m = 0
        for i in range(0,len(best_a)):
            u_m += kp(X_training[i],X_testing[index_sample_test],best_p) * best_a[i] * Y_training[i]
        y_pred[index_sample_test] = np.sign(u_m)

    #write predictions to file
    np.savetxt('kplabel.csv', y_pred, delimiter=',')
    
    print('=========================')
    print('Part 3 Predictions  - kplabel.csv Generated')
    print('=========================')



def Part3(X_training,Y_training,X_validation, Y_validation,X_testing, Polynomial_degree, MAX_ITERATIONS):

    #data
    X = X_training
    Y = Y_training
    #result containers
    accuracy_training = []
    accuracy_validation = []
    #number of training samples
    N = X.shape[0]
    #number of features
    F = X.shape[1]
    #misclassification counter (one for each data point)
    a = np.zeros(N)
    #degree of polynomial used for kernel
    p = Polynomial_degree

    ###############################
    # kernel matrix / gram matrix #
    ###############################
    print('computing kernel matrix')
    K = np.zeros((N,N)).astype(float)
    for i in range(0,N):
        for j in range(0,N):
            K[i][j] = kp(X[i],X[j],p)

    #start iterations
    for iter in range(1,MAX_ITERATIONS+1):
        print('\ntraining for iter: '+str(iter)+'/'+str(MAX_ITERATIONS))
        for index_sample in range(0,N):
            #print('sample index:'+str(index_sample))
            u_m = 0
            for i in range(0,N):
                u_m += K[i][index_sample] * a[i] * Y[i]
            u_m = np.sign(u_m)
            if (Y[index_sample]*u_m) <= 0:
                a[index_sample] += 1

        #training predictions for this iteration train accuracy
        #print('calculating train accuracy for this iter')
        correct_predictions_training = 0
        for index_sample_train in range(0,X.shape[0]):
            u_m = 0
            for i in range(0,N):
                u_m += K[i][index_sample_train] * a[i] * Y[i]
            u_m = np.sign(u_m)
            if (Y[index_sample_train]*u_m) > 0:
                correct_predictions_training += 1
        train_accuracy = 100*(correct_predictions_training/float(Y.shape[0]))
        accuracy_training.append(train_accuracy)
        print('train accuracy:'+str(train_accuracy))
        #validation predictions for this iteration validation accuracy
        #print('calculating validation accuracy for this iter')
        correct_predictions_validation = 0
        for index_sample_val in range(0,X_validation.shape[0]):
            u_m = 0
            for i in range(0,N):
                u_m += kp(X[i],X_validation[index_sample_val],p) * a[i] * Y[i]
            u_m = np.sign(u_m)
            if (Y_validation[index_sample_val]*u_m) > 0:
                correct_predictions_validation += 1
        validation_accuracy = 100*(correct_predictions_validation/float(Y_validation.shape[0]))
        accuracy_validation.append(validation_accuracy)
        print('validation accuracy:'+str(validation_accuracy))

    #get max validation accuracy accross all iterations for this Polynomial_degree
    max_validation_accuracy = max(accuracy_validation)
    #plot train and validation accuracies accross iterations (for this Polynomial_degree)
    plot_Part3(accuracy_training, accuracy_validation, Polynomial_degree, MAX_ITERATIONS)
    #return max_validation_accuracy and alpha vector for recording accross Polynomial_degree
    return max_validation_accuracy, a


def plot_Part3(sse_training, sse_validation, Polynomial_degree, max_iteration ):
        plt.figure()
        plt.plot(sse_training, label = "Training Accuracy",color='blue', linestyle='solid', linewidth = 1.5)
        plt.plot(sse_validation, label = "Validation Accuracy",color='red', linestyle='solid', linewidth = 1.5 )

        plt.xlabel('Iterations (total = ' + str(max_iteration) + ' )', fontsize=12)
        plt.ylabel('Accuracy \n max_train = ' + str(np.amax(sse_training)) + ',  \n max_validation = ' + str(np.amax(sse_validation))  , fontsize=11)

        plt.title('Polynomial Kernel Perceptron (degree=' + str(Polynomial_degree) + '): Accuracy vs Iterations', fontsize=11)
        plt.legend(loc=7)
        plt.grid()
        # changingY-axis range
        x1,x2,y1,y2 = plt.axis()

        plt.savefig("Part3_Accuracy_vs_Iterations_p=" + str(Polynomial_degree) + "_iter=" + str(max_iteration) + "_.png", dpi = 300,bbox_inches="tight")
        plt.show()


def preprocessing(train_csv, dev_csv, test_csv):
    print('=========================')
    print('Preprocessing Running')
    print('=========================')

    #load data from train csv
    data =  np.genfromtxt(train_csv, delimiter=',', dtype=float)
    y = np.zeros_like(data[:,0])
    np.copyto(y, data[:,0])
     #convert to float
    data = np.asarray(data, dtype=float)
    y = np.asarray(y, dtype=float)
    #converting first column into bias
    data[:,0] = np.ones(data.shape[0])


    #load data from validate csv
    data_valid =  np.genfromtxt(dev_csv, delimiter=',', dtype=float)
    #get targets (price)
    y_valid = np.zeros_like(data_valid[:,0])
    np.copyto(y_valid, data_valid[:,0])
    #convert to float
    data_valid = np.asarray(data_valid, dtype=float)
    y_valid = np.asarray(y_valid, dtype=float)
    #converting first column into bias
    data_valid[:,0] = np.ones(data_valid.shape[0])


    #load data from test csv
    data_test =  np.genfromtxt(test_csv, delimiter=',', dtype=float)
    #convert to float
    data_test = np.asarray(data_test, dtype=float)
    #add bias
    bias = np.ones( (data_test.shape[0],1) )
    data_test = np.hstack((bias, data_test))

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


    return [data, y, data_valid, y_valid, data_test]



if __name__ == '__main__':
    main()
