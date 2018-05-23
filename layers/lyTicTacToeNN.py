import numpy as np
import matplotlib.pyplot as plt





def _trainNN(LY_DIM, LY_ACT, X, Y, parameters, costs, learning_rate = 1, num_iter = 2500):
    
    for i in range(0, num_iter):
        a3, cache = forward_propagation(LY_DIM, LY_ACT, X, parameters)
        
        # Loss
        cost = compute_loss(a3, Y)
    
        # Backward propagation.
        grads = backward_propagation(LY_DIM, LY_ACT, X, Y, cache)
        
        # Update parameters.
        parameters = update_parameters(LY_DIM, LY_ACT, parameters, grads, learning_rate)
        
    costs.append(cost)
   
    return parameters, costs
   
def addTrain_movimientos(train_X, train_Y, PARAMS, COST, X, Y):
    #incluir tablero en train_X
    #Opcion 1) incluir como tablero
    #Opcion 2) Incluir jugada a jugada
    numJugadas = np.max(X)
    X1 = train_X
    Y1 = train_Y
    P1 = PARAMS
    C1 = COST
    j1 = np.abs(X)
    for i in range(0,int(numJugadas)):
        
        j2 = j1 <= i + 1
        j3 = X * j2
        j4 = j3.reshape(9,1)
        jugada = np.sign(j4)
        X1 = np.append(X1, jugada, axis=1)
        #lab1 = Y * ((-1)**i) * (10 - numJugadas + i)
        lab1 = (((Y * (10 - numJugadas + i)) / 10)+1)/2
        lab2 = lab1.reshape(1,1)
        Y1 = np.append(Y1, lab2, axis=1)
    P1, C1 = _trainNN(X1, Y1, P1, C1)
    return X1, Y1, P1, C1

def addTrain(LY_DIM, LY_ACT, train_X, train_Y, PARAMS, COST, X, Y):
    #incluir tablero en train_X
    #Opcion 1) incluir como tablero
    #Opcion 2) Incluir jugada a jugada
    numJugadas = np.sum(np.sign(X)==Y)
    X1 = train_X
    Y1 = train_Y
    P1 = PARAMS
    C1 = COST
    j2 = X.reshape(9,1)
    jugada = np.sign(j2)
    X1 = np.append(X1, jugada, axis=1)
    lab1 = (Y*(0.5 + 0.25 * (5-numJugadas)) + 1 )/2
    lab2 = lab1.reshape(1,1)
    Y1 = np.append(Y1, lab2, axis=1)
    P1, C1 = _trainNN(LY_DIM, LY_ACT, X1, Y1, P1, C1)
    return X1, Y1, P1, C1


#print (train_Y.shape)


"""parameters = model(train_X, train_Y, initialization = "he")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
print ("predictions_train = " + str(predictions_train))
print ("predictions_test = " + str(predictions_test))

"""