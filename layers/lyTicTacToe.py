import numpy as np
import random
from nn import NeuralNetwork

def getPrediction(tablero, nn):
    return nn.getPrediction(tablero)

def getNextMove(nn, tablero, move, human):

    player = (-1)**np.sum(tablero!=0)
    if (human == -1):
        j=0
        maxp=-2
        maxj=-1
        while(j<9):
            if (tablero[j]==0):
                tab2 = np.sign(tablero)
                tab2[j] = player 
                thisp = player * getPrediction(tab2, nn) +random.random()*0.1
                if (thisp>maxp):
                    maxp = thisp
                    maxj = j
            j+=1
    else:
        preguntar = True
        while(preguntar):
            print(tablero.reshape(3,3))
            maxj = int(input("movimiento"))
            if (maxj in range(0,tablero.shape[0])):
                preguntar = (tablero[maxj] != 0) 
    tablero[maxj] = player * move
    return

def inicio():
    nn = NeuralNetwork([9, 9, 1], ['', 'T', 'S'])
    
    for j in range(1,1000):
        tablero = np.zeros(9)
        i=0
        human = (-1)**j
        while (i<9):
            human = - human
            if auto:
                human = -1
            getNextMove(nn, tablero, i+1, human)
            end = verify(tablero)
            if (end != 0):
                i=10
            i+=1
        print(tablero.reshape(3,3))
        print(end)
        #input('Next...')
        nn.addTrain(tablero, end)
    return

def verify(tablero):
    tabsgn = np.sign(tablero)
    tab3 = tabsgn.reshape(3,3)
    tab3h = np.sum(tab3, axis=0)
    wonA = np.sum(tab3h==3)
    wonB = np.sum(tab3h==-3)
    if(wonA==wonB):
        tab3v = np.sum(tab3, axis=1)
        wonA = np.sum(tab3v==3)
        wonB = np.sum(tab3v==-3)
    if(wonA==wonB):
        v1 = tabsgn[0] + tabsgn[4] + tabsgn[8]
        v2 = tabsgn[2] + tabsgn[4] + tabsgn[6]
        if (v1==3  or v2==3):
            wonA = 1 
        if (v1==-3 or v2==-3):
            wonB = 1 
    return wonA - wonB 



auto = True

inicio()

