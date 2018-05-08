import numpy as np
import matplotlib.pyplot as plt
from init_utils import compute_loss, forward_propagation, backward_propagation, sigmoid
from init_utils import update_parameters, predict#, load_dataset


def xxx_inicio_prueba():
    X0 = np.array([
    [ 1., -1.], [-1.,  1.], [ 1.,  1.],
    [ 1.,  1.], [ 1.,  1.], [-1., -1.],
    [-1.,  1.], [ 1., -1.], [-1., -1.]
    ])
    Y0 = np.array([[0, 1]])
    Y1 = np.array([[0, 0]])
    
    train_X = X0
    test_X = X0
    train_Y = Y0
    test_Y = Y1
    return train_X, test_X, train_Y, test_Y

def xxx_model(X, Y, learning_rate = 0.001, num_iterations = 1, print_cost = True, initialization = "he"):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
    learning_rate -- learning rate for gradient descent 
    num_iterations -- number of iterations to run gradient descent
    print_cost -- if True, print the cost every 1000 iterations
    initialization -- flag to choose which initialization to use ("zeros","random" or "he")
    
    Returns:
    parameters -- parameters learnt by the model
    """
        
    grads = {}
    costs = [] # to keep track of the loss
    m = X.shape[1] # number of examples
    layers_dims = [X.shape[0], 10, 5, 1]
    
    """    if initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
    """

    # Initialize parameters dictionary.
    parameters = initialize_parameters_he(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        a3, cache = forward_propagation(X, parameters)
        
        # Loss
        cost = compute_loss(a3, Y)

        # Backward propagation.
        grads = backward_propagation(X, Y, cache)
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
            
    # plot the loss
    """    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    """    
    return parameters

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
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2./layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        ### END CODE HERE ###
        
    return parameters

def _initNN():

    costs = [] # to keep track of the loss
    layers_dims = [9, 3, 1, 1]
    # Initialize parameters dictionary.
    parameters = initialize_parameters_he(layers_dims)
    return parameters, costs

def _trainNN(X, Y, parameters, costs, learning_rate = 0.1, num_iter = 25):
    
    for i in range(0, num_iter):
        a3, cache = forward_propagation(X, parameters)
        
        # Loss
        cost = compute_loss(a3, Y)
    
        # Backward propagation.
        grads = backward_propagation(X, Y, cache)
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
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

def addTrain(train_X, train_Y, PARAMS, COST, X, Y):
    #incluir tablero en train_X
    #Opcion 1) incluir como tablero
    #Opcion 2) Incluir jugada a jugada
    numJugadas = np.max(X)
    X1 = train_X
    Y1 = train_Y
    P1 = PARAMS
    C1 = COST
    j2 = X.reshape(9,1)
    jugada = np.sign(j2)
    X1 = np.append(X1, jugada, axis=1)
    lab1 = ((Y * (10 - numJugadas)) / 10)
    lab2 = lab1.reshape(1,1)
    Y1 = np.append(Y1, lab2, axis=1)
    P1, C1 = _trainNN(X1, Y1, P1, C1)
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