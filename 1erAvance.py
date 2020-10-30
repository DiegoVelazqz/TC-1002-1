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