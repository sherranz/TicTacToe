
import numpy as np
import random
import TicTacToeNN as tttnn

def getPrediction(tablero):
    return tttnn.getNNPrediction(tablero)
#    return random.random()*2 -1
    
def getNextMove(tablero, move):

    player = (-1)**np.sum(tablero!=0)

    j=0
    maxp=-1
    maxj=-1
    while(j<9):
        if (tablero[j]==0):
            tab2 = np.sign(tablero)
            tab2[j] = player 
            thisp = player * getPrediction(tab2)
            if (thisp>maxp):
                maxp = thisp
                maxj = j
        j+=1
    tablero[maxj] = player * move
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

def inicio():
    tablero = np.zeros(9)


    i=0
    while (i<9):
        getNextMove(tablero, i+1)
        end = verify(tablero)
        if (end != 0):
            i=10
        i+=1
    #    input('Next...')
    print(tablero.reshape(3,3))
    print(end)
    return

inicio()




