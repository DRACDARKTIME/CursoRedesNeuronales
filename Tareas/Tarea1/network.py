# %load network.py

"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #To improve the weights we need to choose std = 1/sqrt{n_{l-1}} were n_{l-1} is the number of neurons in the layer l-1.
        self.weights = [np.random.randn(y, x)*(1/np.sqrt(x))
                        for x, y in zip(sizes[:-1], sizes[1:])]
        # Debemos de crear los momentos al inicio de la clase dado que 
        # queremos conservar sus valores a lo largo de las epocas de entrenamiento.
        self.M_b = [np.zeros(b.shape) for b in self.biases]  
        self.M_w = [np.zeros(w.shape) for w in self.weights]
        self.R_b = [np.zeros(b.shape) for b in self.biases]  
        self.R_w = [np.zeros(w.shape) for w in self.weights]
        self.r_b = [np.zeros(b.shape) for b in self.biases]  
        self.r_w = [np.zeros(w.shape) for w in self.weights]
        self.m_b = [np.zeros(b.shape) for b in self.biases]  
        self.m_w = [np.zeros(w.shape) for w in self.weights]
        # El tiempo tambien es universal y continua a lo largo de las epocas
        self.t = 0

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input.
        For cross-entropy in the last layer we need softmax and not sigmoid"""
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, a) + b
            if i ==len(self.weights) -1:
                a = softmax(z)
            else:
                a = sigmoid(z)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]  
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)              # Calculamos el gradiente de la funcion de perdida 
                                                                            # para una sola muestra (x,y)
                                                                            # respecto de los parametros w y b.
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]   # Vamos sumando el gradiente en la componente b de 
                                                                            # todas las muestras del minibatch 
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]   # Lo mismo que arriba pero para la componente w.

        self.weights = [w-(eta/len(mini_batch))*nw                          # Actualizamos los pesos haciendo que vayan en
                        for w, nw in zip(self.weights, nabla_w)]            # direccion contraria al gradiente con el paso                                                                  
                                                                            # eta por el promedio de la suma de los gradientes. 

        self.biases = [b-(eta/len(mini_batch))*nb                           # Lo mismo pero para los biases
                       for b, nb in zip(self.biases, nabla_b)]              # Estamos aplicando mini batch stocastic gradient descent
   
    def Adam(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, beta_1 = 0.9, beta_2= 0.999):
        """Train the neural network using Adam. 
        The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch_adam(mini_batch, eta, beta_1, beta_2)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch_adam(self, mini_batch, eta, beta_1, beta_2):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        #inicializamos todos los valores en cero
        nabla_b = [np.zeros(b.shape) for b in self.biases]  
        nabla_w = [np.zeros(w.shape) for w in self.weights] 
        # pondremos epsilon 10^{-8} para que no haya problemas de division entre cero.
        epsilon = 1e-8
        self.t += 1 
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)              # Calculamos el gradiente de la funcion de perdida 
                                                                            # para una sola muestra (x,y)
                                                                            # respecto de los parametros w y b.
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]   # Vamos sumando el gradiente en la componente b de 
                                                                            # todas las muestras del minibatch 
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]   # Lo mismo que arriba pero para la componente w.

        #Promediamos los gradientes
        nabla_w = [(1/len(mini_batch))*nw for nw in nabla_w] 
        nabla_b = [(1/len(mini_batch))*nb for nb in nabla_b]

        # Calculamos M,R, m, r y actualizamos los pesos para cada parametro
        # Las formulas son las vistas en clase.
        # Actualizamos los momentos
        self.M_w = [beta_1*Mw + (1-beta_1)*nw for Mw, nw in zip(self.M_w, nabla_w)]
        self.M_b = [beta_1*Mb + (1-beta_1)*nb for Mb, nb in zip(self.M_b, nabla_b)]
        self.R_w = [beta_2*Rw + (1-beta_2)*nw**2 for Rw, nw in zip(self.R_w, nabla_w)]
        self.R_b = [beta_2*Rb + (1-beta_2)*nb**2 for Rb, nb in zip(self.R_b, nabla_b)]
        # Calculamos las estimaciones con corrección de sesgo principalmente al inicio del codigo
        # dado que los momentos estan sesgados a cero.
        self.m_w = [Mw/(1-beta_1**self.t) for Mw in self.M_w]
        self.m_b = [Mb/(1-beta_1**self.t) for Mb in self.M_b]
        self.r_w = [Rw/(1-beta_2**self.t) for Rw in self.R_w]
        self.r_b = [Rb/(1-beta_2**self.t) for Rb in self.R_b]
        # Actualizamos los parametros w's y b's
        self.weights = [w - eta*(mw)/(np.sqrt(rw)+epsilon) 
                        for w, mw, rw in zip(self.weights, self.m_w, self.r_w)]
        self.biases = [b - eta*(mb)/(np.sqrt(rb)+epsilon)
                        for b, mb, rb in zip(self.biases, self.m_b, self.r_b)]                
    
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, activation) + b
            zs.append(z)
            if i < len(self.weights) - 1:
                activation = sigmoid(z)
            else: # we just changed the last activation in the last layer, we need to use softmax
                activation = softmax(z)

            activations.append(activation)
                                                    #TODO: indicar las dimensiones de z, w, activation y b
                                                    # sea n_l el numero de nueronas en la capa l
                                                    # dim(b^l) = (n_l, 1)
                                                    # dim(w^l) = (n_l, n_{l-1})
                                                    # dim(activation^l) = (n_{l-1}, 1)
                                                    # dim(z^l) = (n_l, 1)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) # (a - y) for cross entropy
        nabla_b[-1] = delta                         #TODO: indicar la dimension de delta
                                                    # Sea L la ultima capa
                                                    # dim(cons_derivate) = dim(z^L) = (n_L, 1)
                                                    # dim(sigmoid_prime(z^L)) = dim(z^L) = (n_L, 1)
                                                    # dim(delta^L) = dim(z^L) = (n_L, 1)
                                                    # En todas las operaciones se hace el producto Hadamard
                                                    # la cual conserva las dimensiones de las matrices.

        nabla_w[-1] = np.dot(delta, activations[-2].transpose())        #TODO: indicar la dimension de nabla_w\
                                                                        # dim(activation^{L-1}) = (n_{L-1}, 1)
                                                                        # dim(nabla_w^{L}) = (n_L,1)x(n_{L-1},1)^T
                                                                        # dim(nabla_w^{L}) = (n_L, n_{L-1})
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z) 
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp      #TODO: indicar la dimension de delta,
                                                                        # self.weights[-l+1].transponse y sp
                                                                        # Aqui usamos l del ciclo for, l=2,3,...
                                                                        # dim(sp) = dim(z^{L-l+2}) = (n_{L-l+2}, 1) 
                                                                        # dim(w^{L-l+2}^T) = (n_{L-l+2}, n_{L-l+2-1} )^T 
                                                                        # dim(w^{L-l+2}^T) = (n_{L-l+1}, n_{L-l+2} ) 
                                                                        # dim(delta^{L-l+1}) = (n_{L-l+1}, n_{L-l+2}) x (n_{L-l+2}, 1)
                                                                        # dim(delta^{L-l+1}) = (n_{L-l+1}, 1)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def softmax(z): # we need to add softmax for cross-entropy
    """
    Numerically stable Softmax.
    softmax(z_i) =
        exp(z_i - max(z)) / 
        sum_j exp(z_j - max(z))
    """
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)
