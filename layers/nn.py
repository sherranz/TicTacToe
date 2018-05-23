import numpy as np
import matplotlib.pyplot as plt
import h5py
import sklearn
import sklearn.datasets
from builtins import NotImplementedError

def sigmoid(x):
    if (np.max(-x)>500):
        x0 = x>-500
        x1 = x*x0
        s = 1/(1+np.exp(-x1))
        s1 = (1 - 1 * x0) * 10**(-4)
        s = s * x0 + s1
    else:
        s = 1/(1+np.exp(-x))
    return s

def relu(x):
    return np.maximum(0,x)

def tanh(x):
    return np.tanh(x)

def dzsigmoid(a, Y, m):
    #Derivada
    # dg/dz = a(1-a)
    return 1./m * (a - Y)

def dsigmoid(x):
    #Derivada
    # dg/dz = a(1-a)
    a = sigmoid(x)
    return a * (1 - a)

def drelu(x):
    #Derivada
    # dg/dz = z>0
    return 1 * (x>0)
    
def dtanh(x):
    #Derivada
    # dg/dz = 1-a^2
    return 1 - np.tanh(x) ** 2

class NeuralNetwork:
    LY_DIM      = [] #Dimensiones de las capas
    LY_ACT      = [] #Funciones de activacion de las capas
    costs       = [] # to keep track of the loss
    parameters  = [] # 
    train_X     = []
    train_Y     = []
        

    
    def initialize_parameters_he(self, layers_dims):
        # np.random.seed(3)
        parameters = {}
        L = len(layers_dims) - 1 # integer representing the number of layers
         
        for l in range(1, L + 1):
            parameters['W' + str(l)] = np.abs(np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2./layers_dims[l-1])) / 100
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
            
        return parameters

    def __init__(self, newLY_DIM, newLY_ACT):
        print ("Instantiating new Neural Network")
        if len(newLY_DIM) != len(newLY_ACT):
            raise AssertionError("Numero de capas distinto a numero de funciones de activacion")
        self.LY_DIM = newLY_DIM
        self.LY_ACT = newLY_ACT
        self.parameters = self.initialize_parameters_he(self.LY_DIM)
        self.train_X = np.zeros((self.LY_DIM[0],0))
        self.train_Y = np.zeros((self.LY_DIM[len(self.LY_DIM) - 1],0))
        return
    
    def forward_propagation(self, X = self.train_X):
        a = X
        cache={}
        for i in range(1, len(self.LY_DIM)):
            W = self.parameters["W" + str(i)]
            b = self.parameters["b" + str(i)]
            z = np.dot(W, a) + b
            actv = self.LY_ACT[i]
            if actv == 'T':
                a = tanh(z)
            elif actv == 'R':
                a = relu(z)
            elif actv == 'S':
                a = sigmoid(z)
            cache["z" + str(i)] =z  
            cache["a" + str(i)] =a
            cache["W" + str(i)] =W
            cache["b" + str(i)] =b  
        return a, cache
    
    def getPrediction(self, X):
        a, _ = self.forward_propagation(X.reshape(self.LY_DIM[0],-1))
        return a
    
    def compute_loss(self, a3, Y = self.train_Y):
        m = Y.shape[1]
        logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
        loss = 1./m * np.nansum(logprobs)
        
        return loss
    
    def update_parameters(self, grads, learning_rate):
        L = len(self.parameters) // 2 # number of layers in the neural networks
        # Update rule for each parameter
        for k in range(L):
            self.parameters["W" + str(k+1)] = self.parameters["W" + str(k+1)] - learning_rate * grads["dW" + str(k+1)]
            self.parameters["b" + str(k+1)] = self.parameters["b" + str(k+1)] - learning_rate * grads["db" + str(k+1)]
            
        return
    
    def trainNN(learning_rate = 1, num_iter = 2500):
        
        for i in range(0, num_iter):
            a, cache    = self.forward_propagation()
            cost        = self.compute_loss(a)
            grads       = self.backward_propagation(cache)
            
            self.update_parameters(grads, learning_rate)
        costs.append(cost)
        return
       
    def addTrain(X, Y):
        #incluir tablero en train_X
        #Opcion 1) incluir como tablero
        #Opcion 2) Incluir jugada a jugada
        numJugadas = np.sum(np.sign(X)==Y)
        j2 = X.reshape(9,1)
        jugada = np.sign(j2)
        self.train_X = np.append(self.train_X, jugada, axis=1)
        lab1 = (Y*(0.5 + 0.25 * (5-numJugadas)) + 1 )/2
        lab2 = lab1.reshape(1,1)
        self.train_Y = np.append(self.train_Y, lab2, axis=1)
        if (self.train_X.shape[1]>4):
            self.trainNN()
        return

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def backward_propagation(self, cache, X = self.train_X):
        m = X.shape[1]
        for i in range (len(self.LY_DIM), 0, -1):
            z = cache['z' + str(i-1)]
            a = cache['a' + str(i-1)]
            W = cache['W' + str(i-1)]
            b = cache['b' + str(i-1)]
            if i>1:
                a_= cache['a' + str(i-2)]
            else:
                a_= X
        
            if LY_ACT[i+1] == 'T':
                dz = dtanh(a, Y, m)
            elif LY_ACT[i+1] == 'R':
                da = np.dot(W.T, dz3)
                dz2 = np.multiply(da2, np.int64(a2 > 0))
            elif LY_ACT[i+1] == 'S':
                dz = dsigmoid(a, Y, m)
            dW =  np.dot(dz3, a2.T)
        """    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache
        
        dz3 = 1./m * (a3 - Y)
        dW3 = np.dot(dz3, a2.T)
        db3 = np.sum(dz3, axis=1, keepdims = True)
        
        da2 = np.dot(W3.T, dz3)
        dz2 = np.multiply(da2, np.int64(a2 > 0))
        dW2 = np.dot(dz2, a1.T)
        db2 = np.sum(dz2, axis=1, keepdims = True)
        
        da1 = np.dot(W2.T, dz2)
        dz1 = np.multiply(da1, np.int64(a1 > 0))
        dW1 = np.dot(dz1, X.T)
        db1 = np.sum(dz1, axis=1, keepdims = True)
        
        gradients = {            "dz3": dz3, "dW3": dW3, "db3": db3,
                     "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                     "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}
        
        """
        return gradients
    
