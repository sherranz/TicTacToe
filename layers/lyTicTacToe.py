import numpy as np
import random
import lyTicTacToeNN as tttnn

def inicio():
    PARAMS, COST, LY_DIM, LY_ACT = tttnn._initNN()
    train_X = np.array(([[],[],[],[],[],[],[],[],[]]))
    #print (train_X.shape)
    train_Y = np.array(([[]]))    
    
    for j in range(1,1000):
        tablero = np.zeros(9)
    
    
        i=0
        human = (-1)**j
        while (i<9):
            human = - human
            getNextMove(LY_DIM, LY_ACT, tablero, i+1, PARAMS, human)
            end = verify(tablero)
            if (end != 0):
                i=10
            i+=1
        #    input('Next...')
        print(tablero.reshape(3,3))
        print(end)
        train_X, train_Y, PARAMS, COST = tttnn.addTrain(LY_DIM, LY_ACT, train_X, train_Y, PARAMS, COST, tablero, end)
    return

def getPrediction(tablero, parameters):
    return tttnn.getNNPrediction(tablero, parameters)
#    return random.random()*2 -1
    
def getNextMove(tablero, move, parameters, human):

    player = (-1)**np.sum(tablero!=0)
    if (human == -1):
        j=0
        maxp=-2
        maxj=-1
        while(j<9):
            if (tablero[j]==0):
                tab2 = np.sign(tablero)
                tab2[j] = player 
                thisp = player * getPrediction(tab2, parameters) +random.random()*0.1
                if (thisp>maxp):
                    maxp = thisp
                    maxj = j
            j+=1
    else:
        preguntar = True
        while(preguntar):
            print(tablero.reshape(3,3))
            maxj = int(input("movimiento"))
            preguntar = (tablero[maxj] != 0) 
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





inicio()

