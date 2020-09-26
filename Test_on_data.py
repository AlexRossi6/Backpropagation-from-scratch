from Backpropagation import Backpropagation
import numpy as np
import PIL
import matplotlib.pyplot as plt
from useful_functions import evaluation

#preprocessing data
X=[]
shapes = ['circles','squares','triangles']
for j in shapes:
    for i in range(1,101):
    
        img = PIL.Image.open('shapes\{}\drawing({}).png'.format(j,i)).convert('1')
        imgarr = np.array(img)
        X.append(imgarr)



X = np.array(X)
X = np.reshape(X,(300,784))
Y =[]
y_poss = [[1,0,0],[0,1,0],[0,0,1]]

for i in [0,1,2]:
    for j in range(100):
        Y.append(y_poss[i])
        


#analysis

w,b,cost_history = Backpropagation([784,500,300,100,30,3],0.3,2000,X,Y,100)

plt.figure(figsize=(10,5))
plt.plot(cost_history)
plt.show()

result = evaluation(w,b,X,Y)

print('Percentage of true classifications: {:.2f}%'.format(result*100))





