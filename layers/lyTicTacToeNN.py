import numpy as np
import matplotlib.pyplot as plt
from lyinit_utils import compute_loss, forward_propagation, backward_propagation, sigmoid
from lyinit_utils import update_parameters, predict#, load_dataset

def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers
     
    for l in range(1, L + 1):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.abs(np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2./layers_dims[l-1]) / 100)
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        ### END CODE HERE ###
        
    return parameters

def _initNN():

    costs = [] # to keep track of the loss
    layers_dims = [9, 3, 1]
    layers_actv = ['', 'T', 'S']
    # Initialize parameters dictionary.
    parameters = initialize_parameters_he(layers_dims)
    return parameters, costs, layers_dims, layers_actv

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
   
def getNNPrediction(X, PARAMS):
    a3, caches = forward_propagation(X.reshape((9,-1)), PARAMS)
    return a3

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