import numpy as np
import useful_functions as us



#str = structure of the NN (list of number of neurons for each layer)
#m = size of the mini-batch
def Backpropagation(str,lr,epoch,X,Y,m):
    X = np.array(X)
    Y = np.array(Y)
    #L = number of layers
    L = len(str)
    w = [np.random.randn(str[i], str[i+1]) for i in range(0, L - 1)]
    b = [np.random.randn(str[i + 1]) for i in range(0, L - 1)]
    a = [np.zeros(str[i]) for i in range(0, L)]
    z = [np.zeros(str[i]) for i in range(1, L)]
    #e = vector of errors for each neuron at each layer
    e = [np.zeros(str[i]) for i in range(1,L)]
    
    cost_history = []
    
    for i in range(0, epoch):
        print('epoch: ',i)
        
        cost = 0
        #random shuffle of data to randomly select a mini batch 
        X, Y = us.shuffle_data(X, Y)
        x_mini_batch = X[0:m, ]
        y_mini_batch = Y[0:m, ]
        
        #'accumulator' of the error e
        acc_e = np.array([np.zeros(str[i]) for i in range(1, L)])

        for x,y in zip(x_mini_batch,y_mini_batch):
            #feedforward
            a[0] = x
            z,a = us.feedforward(z,a,w,b,L)

            #Backpropagation of the error
            e,acc_e = us.backprop_err(e,a,y,z,L,acc_e,w)

            cost += np.sum((y - a[L - 1]) ** 2)
            
        #update weights and biases
        b,w = us.update(b,w,lr,m,acc_e,a)
        
        print('cost: ',cost/m)
        

        cost_history.append(cost/m)
        
    return w,b,cost_history



