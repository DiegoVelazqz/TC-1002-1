import pandas as pd
import numpy as np
import math
import operator

def dE(datos1, datos2, leng):
    dist = 0
    for i in range(leng):
        dist += np.square(datos1[i] - datos2[i])
    return np.sqrt(dist)

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


def multknn(dfTraining, dfPredic,k):
    y=[]
    ll=dfPredic.values.tolist()
    
    for i in range(dfPredic.shape[0]):
        dato=pd.DataFrame(ll[i])
        result,neigh = knn(dfTraining, dato, k)
        y.append(result)
    return y

dat = pd.read_csv('data/iris.csv')
testSet = [[7.2, 3.6, 5.1, 2.5]]
test = pd.DataFrame(testSet)


k=int(input('Con cuántos puntos cercanos quieres hacer la predicción: '))

result,neigh = knn(dat, test, k) 

print('\n\nCon ',k,' Vecinos Cercanos')
print('\nClase predicha del punto = ',result)
print('Vecinos más cercanos del punto = ',neigh)

test=pd.read_csv('data/iris2.csv')
yhat = multknn(dat, test, 1)
df=pd.read_csv('data/resultados.csv')

y2=df.iloc[:,-1].values.tolist()
y2=np.matrix(y2)
yhat=np.matrix(yhat)
print(y2==yhat)
comp=(y2==yhat)*1
tot=np.sum(comp)
