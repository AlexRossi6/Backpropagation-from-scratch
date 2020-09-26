import numpy as np


def sigmoid(z):
    
    return (1 / (1 + np.exp(-z)))



def sigmoid_derivative(z):
    
    return (sigmoid(z)*(1-sigmoid(z)))



def feedforward(z,a,w,b,L):
    for l in range(0,L-1):
        z[l] = np.dot(a[l],w[l]) + b[l]
        a[l+1] = sigmoid(z[l])

    return z,a


def backprop_err(e,a,y,z,L,acc_e,w):
    e[L-2] = (a[L-1] - y) * sigmoid_derivative(z[L-2])
    acc_e[L-2] = acc_e[L-2] + e[L-2]
            
    for l in range(L-3,-1,-1):
        e[l] = np.dot(w[l+1],e[l+1]) * sigmoid_derivative(z[l])
        acc_e[l] = acc_e[l] + e[l]
    
    return e,acc_e


def update(b,w,lr,m,acc_e,a):
    L = len(w)+1
    for l in range(0,L-1):
        b[l] = b[l] - (lr/m)*acc_e[l]
        w[l] = w[l] - (lr/m)* np.dot(a[l].reshape(-1,1),acc_e[l].reshape(1,-1))
    return b,w



def shuffle_data(a, b):
    
    size = len(a)
    indeces = np.random.permutation(size)
    return a[indeces], b[indeces]




def evaluation(w,b,X_test,Y_test):
    L = len(w)+1
    n = X_test.shape[0]
    count = 0
    for x,y in zip(X_test,Y_test):
        a = x
        for l in range(0,L-1):

            z = np.dot(a,w[l]) + b[l]
                
            a = sigmoid(z)
        
        if np.argmax(a)==np.argmax(y):
            
            count += 1
    return count/n
        
    
    
    
    
    

