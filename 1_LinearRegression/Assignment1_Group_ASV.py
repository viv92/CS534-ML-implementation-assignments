import numpy as np
import math
import matplotlib.pyplot as plt
from io import StringIO
import csv




def main():

    X_normalized_training,Y_normalized_training,X_normalized_validation, Y_normalized_validation,X_normalized_testing,X_nonnormalized_training,X_nonnormalized_validation,X_nonnormalized_testing   = preprocessing ('PA1_train.csv','PA1_dev.csv','PA1_test.csv')

    Y_nonnormalized_training = Y_normalized_training
    Y_nonnormalized_validation = Y_normalized_validation

    # PART 1 OF THE ASSIGNMENT
    Part1(X_normalized_training, Y_normalized_training, X_normalized_validation, Y_normalized_validation, 'normalized')\
    # PART 2 OF THE ASSIGNMENT
    W_training =  part2(X_normalized_training, Y_normalized_training, X_normalized_validation, Y_normalized_validation)
#   PART 3 OF THE ASSIGNMENT
    Part3(X_nonnormalized_training, Y_nonnormalized_training, X_nonnormalized_validation, Y_nonnormalized_validation, 'unnormalized')
    Part3(X_normalized_training, Y_normalized_training, X_normalized_validation, Y_normalized_validation, 'normalized')
    # Predictions for TESTING DATA
    predict_on_test(X_normalized_testing, W_training)


def preprocessing(train_csv, dev_csv, test_csv):
    #load data from csv
    data =  np.genfromtxt(train_csv, delimiter=',', dtype=None)
    #remove first row (header)
    data = data[1:, :]
    #print ("@@@1data.shape:", data.shape)
    #remove id column - just a serial number. No correlation with target (price)
    data = np.delete(data, 1, 1)
    #print ("@@@@2data.shape:", data.shape)
    #split dates
    dates = data[:,1]
    months = np.zeros(dates.shape).astype('int')
    days = np.zeros(dates.shape).astype('int')
    years = np.zeros(dates.shape).astype('int')
    for i in range(0, len(dates)):
        date_split = dates[i].split(b'/')
        months[i] = date_split[0]
        days[i] = date_split[1]
        years[i] = date_split[2]
    #add month, day and year columns
    months.shape = (months.shape[0],1)
    days.shape = (days.shape[0],1)
    years.shape = (years.shape[0],1)
    data = np.hstack((data[:,0:1], months, days, years, data[:,2:]))
    #print ("@@@@3data.shape:", data.shape)

    #get targets (price)
    y = data[:,data.shape[1]-1]

    #print ("@@@@@@data.shape:", data.shape)

    #load data from csv
    datav =  np.genfromtxt(dev_csv, delimiter=',', dtype=None)
    #remove first row (header)
    datav = datav[1:, :]
    #remove id column - just a serial number. No correlation with target (price)
    datav = np.delete(datav, 1, 1)
    #split dates
    dates = datav[:,1]
    months = np.zeros(dates.shape).astype('int')
    days = np.zeros(dates.shape).astype('int')
    years = np.zeros(dates.shape).astype('int')
    for i in range(0, len(dates)):
        date_split = dates[i].split(b'/')
        months[i] = date_split[0]
        days[i] = date_split[1]
        years[i] = date_split[2]
    #add month, day and year columns
    months.shape = (months.shape[0],1)
    days.shape = (days.shape[0],1)
    years.shape = (years.shape[0],1)
    datav = np.hstack((datav[:,0:1], months, days, years, datav[:,2:]))

    #get targets (price)
    yv = datav[:,datav.shape[1]-1]


    #load data from csv
    datat =  np.genfromtxt(test_csv, delimiter=',', dtype=None)
    #remove first row (header)
    datat = datat[1:, :]
    #remove id column - just a serial number. No correlation with target (price)
    datat = np.delete(datat, 1, 1)
    #split dates
    dates = datat[:,1]
    months = np.zeros(dates.shape).astype('int')
    days = np.zeros(dates.shape).astype('int')
    years = np.zeros(dates.shape).astype('int')
    for i in range(0, len(dates)):
        date_split = dates[i].split(b'/')
        months[i] = date_split[0]
        days[i] = date_split[1]
        years[i] = date_split[2]
    #add month, day and year columns
    months.shape = (months.shape[0],1)
    days.shape = (days.shape[0],1)
    years.shape = (years.shape[0],1)
    datat = np.hstack((datat[:,0:1], months, days, years, datat[:,2:]))

    #feature stats table
    featstats = np.zeros((4, data.shape[1]))

    for colindex in range(0, featstats.shape[1]-1):
        sum = 0
        for ele in data[:,colindex]:
            sum += float(ele)
        mean = sum/(data.shape[0])
        sum = 0
        for ele in data[:,colindex]:
            sum += (math.pow((float(ele) - mean),2))
        stddev = math.sqrt(sum/(data.shape[0]))
        max = float(data[0][colindex])
        min = float(data[0][colindex])
        for ele in data[:,colindex]:
            if max < float(ele):
                max = float(ele)
            if min > float(ele):
                min = float(ele)
        rangee = max-min
        featstats[0,colindex] = mean
        featstats[1,colindex] = stddev
        featstats[2,colindex] = rangee
        featstats[3,colindex] = min

    #copy for un-normalized part3
    u_data = np.zeros_like(data)
    u_datav = np.zeros_like(datav)
    u_datat = np.zeros_like(datat)
    np.copyto(u_data, data)
    np.copyto(u_datav, datav)
    np.copyto(u_datat, datat)


    #normalize the data
    r_ind = 0
    c_ind = 1
    while r_ind < data.shape[0]:
        c_ind = 1
        while c_ind < data.shape[1]-1:
            data[r_ind][c_ind] = (float(data[r_ind][c_ind])-featstats[3][c_ind])/featstats[2][c_ind]
            c_ind += 1
        r_ind += 1

    r_ind = 0
    c_ind = 1
    while r_ind < datav.shape[0]:
        c_ind = 1
        while c_ind < datav.shape[1]-1:
            datav[r_ind][c_ind] = (float(datav[r_ind][c_ind])-featstats[3][c_ind])/featstats[2][c_ind]
            c_ind += 1
        r_ind += 1

    r_ind = 0
    c_ind = 1
    while r_ind < datat.shape[0]:
        c_ind = 1
        while c_ind < datat.shape[1]:
            datat[r_ind][c_ind] = (float(datat[r_ind][c_ind])-featstats[3][c_ind])/featstats[2][c_ind]
            c_ind += 1
        r_ind += 1

    #convert to float
    data = np.asarray(data, dtype=float)
    datav = np.asarray(datav, dtype=float)
    datat = np.asarray(datat, dtype=float)
    y = np.asarray(y, dtype=float)
    yv = np.asarray(yv, dtype=float)
    u_data = np.asarray(u_data, dtype=float)
    u_datav = np.asarray(u_datav, dtype=float)
    u_datat = np.asarray(u_datat, dtype=float)

    #delete price column
    data = np.delete(data, data.shape[1]-1, 1)
    #delete price column
    datav = np.delete(datav, datav.shape[1]-1, 1)
    #delete price column
    u_data = np.delete(u_data, u_data.shape[1]-1, 1)
    #delete price column
    u_datav = np.delete(u_datav, u_datav.shape[1]-1, 1)

    #return the 5 normalized matrices
    return [data, y, datav, yv, datat, u_data, u_datav, u_datat]



def Part1 (X_training, Y_training, X_validation, Y_validation, text):

    print('Running Part 1 for', text, 'data');
    learning_rates = [1, pow(10,-1), pow(10, -2), pow(10, -3), pow(10, -4), pow(10, -5), pow(10, -6), pow(10, -7)]
    lamda = 0

    writer = csv.writer(open("Part3_" + text + ".csv","w",newline=''), delimiter=',',quoting=csv.QUOTE_ALL)
    for learning_rate in learning_rates:
        print('Running for Learning Rate = ', learning_rate)
        #train for 1000 iterations will get sse_training and sse_validation

        SSE_training, SSE_validation, W_training, gradient, iterations = gradientDescent_Part1(X_training, Y_training,X_validation, Y_validation,learning_rate, lamda)

        writer.writerow( ["SSE_"+text+"_training", "learning rate", learning_rate, SSE_training ] )
        writer.writerow( ["SSE_"+text+"_validation", "learning rate", learning_rate, SSE_validation ] )
        writer.writerow( ["Gradient"+text+"_validation", "learning rate", learning_rate, gradient ] )
        writer.writerow( ["Coefficients"+text+"_validation", "learning rate", learning_rate, W_training ] )
        plot_sse_Part1(SSE_training, SSE_validation, learning_rate, gradient, text, iterations)
    return W_training;
def gradientDescent_Part1(X, y, X2,y2,alpha, lamb):


    converged = False
    #iter = 0
    m = X.shape[0] #Number of samples
    m2 = X2.shape[0]
    #initial w
    w = np.ones(X.shape[1]) * 0
    grad = np.zeros_like(w)

    #total error, J(w)
    sse_X = []
    sse_X2 = []
    gradient = []

    #Iterate loop
    #while not converged:

    xw_y = np.matmul(X,w) - y
    w_reg = w
    w_reg[0] = 0
#    last_gradient = (2*np.matmul(np.transpose(X),xw_y) + 2*lamb*w_reg)
    last_gradient = 0
    iterations = 0

    while 1:


        #reset gradients
        grad = np.zeros_like(w)

        #For each training sample, compute the gradient d/d_theta j(theta)
        xw_y = np.matmul(X,w) - y
        w_reg = w
        w_reg[0] = 0
        grad = (2*np.matmul(np.transpose(X),xw_y) + 2*lamb*w_reg)
        #convergence criterion
#        print(last_gradient)
#        print(grad)
        if (abs(np.linalg.norm(last_gradient) - np.linalg.norm(grad)) < 0.05):
            break
        else:
            last_gradient = grad

        #update the weights
        w -= (alpha * grad)

        #mean squared error
        sse_X.append(np.matmul(np.transpose(xw_y),xw_y))

        xw_y2 = np.matmul(X2,w) - y2
        sse_X2.append(np.matmul(np.transpose(xw_y2),xw_y2))

        normgradient = np.linalg.norm(grad)
        gradient.append(normgradient)


        #divergence criterion

        if (iterations == 5):
            initial_sse = sse_X[iterations]
        if(iterations >5):
            if (  sse_X[iterations] - initial_sse > pow(10,3)):
                break

        iterations += 1

    return [sse_X,sse_X2,w, gradient, iterations]


def plot_sse_Part1(sse_training, sse_validation, learning_rate, gradient, text, iterations):
        #plt.figure(num=None, figsize=(18, 9), dpi=300, facecolor='w', edgecolor='k')
        plt.plot(sse_training, label = text+" Training SSE",color='blue', linestyle='solid', linewidth = 1.5 )
        plt.plot(sse_validation, label = text+" Validation SSE",color='red', linestyle='dashed', linewidth = 1.5 )

        plt.xlabel('Iterations (total = ' + str(iterations) + ' )', fontsize=12)
        plt.ylabel('SSE \n min_train = ' + str(sse_training[-1]) + ',  \n min_validation = ' + str(sse_validation[-1])  , fontsize=11)

        plt.title('SSE vs Iterations for Learning Rate = ' + str(learning_rate) + '\n ', fontsize=11)
        plt.legend(loc=1)

        plt.savefig("Part1_"+text+"_test_validation_sse_lr_"+str(learning_rate) + ".png", dpi = 300,bbox_inches="tight")
        plt.show()

        #plot gradient
        #plt.figure(num=None, figsize=(18, 9), dpi=300, facecolor='w', edgecolor='k')
        plt.plot(gradient,label = text+" Gradient",color='black', linestyle='solid', linewidth = 1.5 )
        plt.xlabel('Iterations (total = ' + str(iterations) + ' )', fontsize=12)
        plt.ylabel('Gradient', fontsize=16)

        plt.title('Gradient vs Iterations for Learning Rate = ' + str(learning_rate), fontsize=11)
        plt.legend(loc=1)

        plt.savefig("Part1_"+text+"_gradient_lr_"+str(learning_rate) + ".png", dpi = 300,bbox_inches="tight")
        plt.show()


def part2(X_tr, y_tr, X_cv, y_dev):
        print ("STARTING PART 2")
        tr_data_float = X_tr
        dev_data_float = X_cv

        alpha = 0.00001
        iter = 10000
        lamb_array = [0,0.001,0.01,0.1,1,10,100]
        tr_loss = []
        dev_loss = []
        for lamb in lamb_array:
            print ("\nProcessing for lambda = ", lamb)
            J_tr,w = gradientDescent_p2(tr_data_float, y_tr, alpha, iter, lamb)
            print ("weights: ")
            print (w)
            tr_loss.append(J_tr[-1])

            #calculate cross val loss
            X_dev = dev_data_float
            xw_y_dev = np.matmul(X_dev,w) - y_dev
            dev_loss.append(np.matmul(np.transpose(xw_y_dev),xw_y_dev))
            #print "iter:", i, "loss: ", J[i]

        #plot
        # fig, ax = plt.subplots()
        # ax.plot(lamb_array, tr_loss, 'r')
        # ax.plot(lamb_array, dev_loss, 'b')
        # ax.set_xscale('log')
        # #ax.set_ylim([0,10])
        # plt.show()

        fig, ax = plt.subplots()
        ax.plot(lamb_array, tr_loss, 'r', label='SSE_train', marker='*')
        ax.plot(lamb_array, dev_loss, 'b', label='SSE_val', marker='*')
        ax.set_xscale('log')
        #ax.set_ylim([0,10])
        plt.title("Tuning regularization parameter 'lambda'")
        plt.legend(loc='upper left')
        plt.xlabel("lambda")
        plt.ylabel("SSE")
        fig.savefig('PART2_crossVal'+'_alpha_'+str(alpha)+'_iter_'+str(iter)+'.png')
        plt.show()

        print ("lamb_array:")
        print (lamb_array)
        print ("tr_loss:")
        print (tr_loss)
        print ("dev_loss")
        print (dev_loss)
        J_tr,w = gradientDescent_p2(tr_data_float, y_tr, pow(10,-5), 65000, 0.1)
        return w

def gradientDescent_p2(X, y, alpha, iterations, lamb):
    converged = False
    #iter = 0
    m = X.shape[0] #Number of samples
    #print "M: ", m

    #initial w
    w = np.ones(X.shape[1]) * 1
    grad = np.zeros_like(w)

    #total error, J(w)
    J = []

    #Iterate loop
    #while not converged:
    for i in range(iterations):

        #reset gradients
        grad = np.zeros_like(w)

        #For each training sample, compute the gradient d/d_theta j(theta)
        xw_y = np.matmul(X,w) - y
        #print "xw_y: ", xw_y
        w_reg = w
        w_reg[0] = 0
        grad = ((2*np.matmul(np.transpose(X),xw_y) + 2*lamb*w_reg))

        #update the weights
        w -= (alpha * grad)

        #mean squared error
        xw_y = np.matmul(X,w) - y
        J.append((np.matmul(np.transpose(xw_y),xw_y)))
        #print "iter:", i, " gradient norm: ", math.sqrt(np.matmul(np.transpose(grad),grad)), " loss: ", J[i]
        #print " predicted price: ", np.matmul(X,w)
    return [J,w]






def Part3 (X_training, Y_training, X_validation, Y_validation, text):
    print('Running Part 3 for', text, 'data');
    learning_rates = [1, 0, pow(10,-3), pow(10, -6), pow(10, -9), pow(10, -15), pow(10, -16)]
    iterations = 10000
    lamda = 0

    writer = csv.writer(open("Part3_" + text + ".csv","w",newline=''), delimiter=',',quoting=csv.QUOTE_ALL)
    for learning_rate in learning_rates:
        print('Running for Learning Rate = ', learning_rate)
        #train for 1000 iterations will get sse_training and sse_validation
        SSE_training, SSE_validation, W_training, gradient = gradientDescent_Part3(X_training, Y_training,X_validation, Y_validation,learning_rate, iterations, lamda)

        writer.writerow( ["SSE_"+text+"_training", "learning rate", learning_rate, SSE_training ] )
        writer.writerow( ["SSE_"+text+"_validation", "learning rate", learning_rate, SSE_validation ] )
        writer.writerow( ["Gradient"+text+"_validation", "learning rate", learning_rate, gradient ] )
        writer.writerow( ["Coefficients"+text+"_validation", "learning rate", learning_rate, W_training ] )
        plot_sse_part3(SSE_training, SSE_validation, learning_rate, gradient, text)

def gradientDescent_Part3(X, y, X2,y2,alpha, iterations, lamb):
    converged = False
    #iter = 0
    m = X.shape[0] #Number of samples
    m2 = X2.shape[0]
    #initial w
    w = np.ones(X.shape[1]) * 0
    grad = np.zeros_like(w)

    #total error, J(w)
    sse_X = []
    sse_X2 = []
    gradient = []

    #Iterate loop
    #while not converged:
    for i in range(iterations):

        #reset gradients
        grad = np.zeros_like(w)

        #For each training sample, compute the gradient d/d_theta j(theta)
        xw_y = np.matmul(X,w) - y
        w_reg = w
        w_reg[0] = 0
        grad = (2*np.matmul(np.transpose(X),xw_y) + 2*lamb*w_reg)

        #update the weights
        w -= (alpha * grad)

        #mean squared error
        sse_X.append(np.matmul(np.transpose(xw_y),xw_y))

        xw_y2 = np.matmul(X2,w) - y2
        sse_X2.append(np.matmul(np.transpose(xw_y2),xw_y2))

        normgradient = np.linalg.norm(grad)
        gradient.append(normgradient)

        #divergence criterion
        if (i == 5):
            initial_sse = sse_X[i]
        if(i>5):
            if (  sse_X[i] - initial_sse > pow(10,3)):
                break


    return [sse_X,sse_X2,w, gradient]


def plot_sse_part3(sse_training, sse_validation, learning_rate, gradient, text):
        #plt.figure(num=None, figsize=(18, 9), dpi=300, facecolor='w', edgecolor='k')
        plt.plot(sse_training, label = text+" Training SSE",color='blue', linestyle='solid', linewidth = 1.5 )
        plt.plot(sse_validation, label = text+" Validation SSE",color='red', linestyle='dashed', linewidth = 1.5 )

        plt.xlabel('Iterations', fontsize=11)
        plt.ylabel('SSE \n min_train = ' + str(sse_training[-1]) + ',  \n min_validation = ' + str(sse_validation[-1])  , fontsize=11)


        plt.title('SSE vs Iterations for Learning Rate = ' + str(learning_rate), fontsize=11)
        plt.legend(loc=1)

        plt.savefig("Part3_"+text+"_test_validation_sse_lr_"+str(learning_rate) + ".png", dpi = 300,bbox_inches="tight")
        plt.show()

        #plot gradient
        #plt.figure(num=None, figsize=(18, 9), dpi=300, facecolor='w', edgecolor='k')
        plt.plot(gradient,label = text+" Gradient",color='black', linestyle='solid', linewidth = 1.5 )
        plt.xlabel('Iterations', fontsize=16)
        plt.ylabel('Gradient', fontsize=16)

        plt.title('Gradient vs Iterations for Learning Rate = ' + str(learning_rate), fontsize=11)
        plt.legend(loc=1)

        plt.savefig("Part3_"+text+"_gradient_lr_"+str(learning_rate) + ".png", dpi = 300,bbox_inches="tight")
        plt.show()
def predict_on_test(data, w):
    print ("-----------------------")



    predictions = np.matmul(data,w)

    A = np.squeeze(np.asarray(predictions))

    np.savetxt("PREDICITION_FILE.csv", A, delimiter=",")



    print('\n Predictions of Test Data are written on PREDICITION_FILE.csv')

if __name__ == '__main__':
    main()
