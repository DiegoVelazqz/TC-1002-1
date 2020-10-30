#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:21:06 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import operator
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.decomposition import FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis
import random

def randomPoint():
    newPoint = []
    num = random.choice(range(-1000, 1000))
    newNum = num/10
    newPoint.append(newNum)
        
    num = random.choice(range(-500, 500))
    newNum = num/10
    newPoint.append(newNum)
    return(newPoint)

def dE(datos1, datos2, leng):
    dist = 0
    for i in range(leng):
        dist += np.square(datos1[i] - datos2[i])
    return np.sqrt(dist)

def multknn(dfTraining, dfPredic,k):
    y=[]
    ll=dfPredic.values.tolist()
    
    for i in range(dfPredic.shape[0]):
        dato=pd.DataFrame(ll[i])
        result,neigh = knn(dfTraining, dato, k)
        y.append(result)
    return y

def knn(trainingSet, inst, k):
    dists = {}
    leng = inst.shape[1]
    
    #Calculo de la distancia euclideana entre cada fila de entrenamiento y de prueba
    for i in range(len(trainingSet)):
        dist = dE(inst, trainingSet.iloc[i], leng)
        dists[i] = dist[0]

    # Ordenando de menor a mayor en cuanto a distancia
    ordenDist = sorted(dists.items(), key=operator.itemgetter(1))
    neighbors = []
    
    # Extraemos los k vecinos más cercanos
    for i in range(k):
        neighbors.append(ordenDist[i][0])
    classVotes = {}
    
    # Calculando la clase que más se repite en los vecinos
    for i in range(len(neighbors)):
        resp = trainingSet.iloc[neighbors[i]][len(trainingSet.columns)-1]
        
        if resp in classVotes:
            classVotes[resp] += 1
        else:
            classVotes[resp] = 1

    ordenVotos = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return(ordenVotos[0][0], neighbors)

sp=pd.read_csv('/Users/diegovelazquez/Downloads/iris.csv')
sp['Tipo_Flor']=sp['Tipo_Flor'].replace(['Iris-versicolor','Iris-virginica','Iris-setosa'],[0,1,2])
data=sp.values
X= data[:,0:-1]
y=data[:,-1]

emb= FactorAnalysis(n_components=2)
X1t =emb.fit_transform(X,y)
plt.scatter(X1t[:,0],X1t[:,-1],c=y)
plt.title('Iris dataset FactorAnalysis')
plt.show()

emb= LinearDiscriminantAnalysis(n_components=2)
X2t =emb.fit_transform(X,y)
plt.scatter(X2t[:,0],X2t[:,-1],c=y)
plt.title('Iris dataset LinearDiscriminant')
plt.show()

emb=NeighborhoodComponentsAnalysis(n_components=2)
X3t =emb.fit_transform(X,y)
plt.scatter(X3t[:,0],X3t[:,-1],c=y)   
plt.title('Iris dataset Neighborhood')
plt.show()

emb= Isomap(n_components=2)
X4t =emb.fit_transform(X,y)
plt.scatter(X4t[:,0],X4t[:,-1],c=y)   
plt.title('Iris dataset Isomap')
plt.show()


emb=MDS(n_components=2)
x5t= emb.fit_transform(X,y)
plt.scatter(x5t[:,0],x5t[:,1],c=y)
plt.title('Iris dataset MDS')
plt.show()

Xn=np.array([randomPoint()])
print(Xn)
punto = pd.DataFrame(Xn)
data=pd.DataFrame(X3t)
data[2]=y
k=int(input('Con cuántos puntos cercanos quieres hacer la predicción: '))
result,neigh = knn(data, punto, k)

if result==0.0:
    tip='Iris versicolor'
elif result==1.0:
    tip='Iris virginica'
elif result==2.0:
    tip='Iris setosa'
else:
    tip='N/A'

print('\n\nCon ',k,' Vecinos Cercanos')
print('\nClase predicha del punto = ',tip)
print('Vecinos más cercanos del punto = ',neigh)

puntoscercanos=pd.DataFrame(columns=data.columns)
for i in range(len(neigh)):
    n=data.iloc[[neigh[i]]]
    puntoscercanos=puntoscercanos.append(n)

pC=puntoscercanos.to_numpy()
plt.scatter(X3t[:,0],X3t[:,-1],c=y)   
plt.scatter(pC[:,0],pC[:,1],c='lime',marker='x',linewidths=(1.5))
plt.scatter(Xn[:,0],Xn[:,1],c='red')
plt.title('KNN')
plt.show()