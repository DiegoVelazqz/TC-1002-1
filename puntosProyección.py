import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from random import randrange as rr
import random

def genData(n,seed=1,m=False): #Genera data linearmente separable por una pendiente (m)
    np.random.seed(seed)
    if(m==False):
    	m= (rr(20)+1)/10
    X = np.random.random((n, 2))
    y = ((X[:,1]/X[:,0])>m)*1
    X = (X-0.5)*2
    return [X,y]

def graphPoints(X,y): #Grafica los puntos y los puntos proyectados
    for i in range(0,X.shape[0]):
        p = X[i,0:2]
        #p2 = Xp[i,0:2]            
        if(y[i]):
            plt.scatter(p[0],p[1],color="red", s=4)
        else:
            plt.scatter(p[0],p[1],color="black", s=4)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()
    
def randomVector():
    v=np.random.randint(-10,high=10,size=2)
    return v 

def randomListVector(n):
    vlist=[]
    for i in range(n):
        v=randomVector()
        vlist.append(v)
    return vlist

def drawVector(lv): #list of vectors
    maxV = 0
    colors = ['black', 'red', 'green', 'orange', 'grey', 'purple', 'brown', 'purple']
    for i in range(len(lv)):
        v = lv[i]  
        plt.quiver(0, 0, v[0], v[1], color=colors[i%len(colors)], angles='xy', scale_units='xy', scale=1)
        if(max(v) > maxV):
            maxV = max(v)
    plt.xlim(-maxV*1.1, maxV*1.1)
    plt.ylim(-maxV*1.1, maxV*1.1)
    plt.show()
    
def proyectPoints(dat,vec):
    l=[]
    for i in range(len(dat[1])):
        v2=(np.dot(dat[0][i],vec[0])/np.linalg.norm(vec[0])**2)*vec[0]
        l.append(v2)
    return l

print('Número de Datos:')
n=int(input())
d=genData(n)
graphPoints(d[0],d[1])
lv=randomListVector(1)
drawVector(lv)
print(d[0][0])
pp=np.array(proyectPoints(d, lv))
graphPoints(pp, d[1])
