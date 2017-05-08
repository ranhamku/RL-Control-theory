# the learning phase of the algorithm will continue
# until training causes one of the actor network weights 
# to exceed its perturbation range. If this never happens, 
# then the algorithm would proceed indefinitely. To prevent
# an infinite loop situation, I try to use additional halting 
# criteria. Here, this takes the form of a fixed number of 
# training trials; after we train for the maximum number of samples
# it exits the learning phase


from numpy import *
import numpy as np

# the epsilon-greedy algorithm uses the actor network output with 
# probability 1 - epsilon or adds a small random perturbation to the 
# actor network output with probability epsilon
Action = [aO,Add_P]        # aO denotes using the actor network output 
                           # and Add_p adding a small random perturbation 
epsilon = 0.9
alpha = 0.1                # learning rate
gamma = 0.9
class CriticNet()
    def __init__(self,)
        # critic weights randomly
        self.qweights = random.random

        self.Bp = np.zeros()          # computed in the stability phase
       
        # the halting criteria
        self.c = np.zeros()
        
        # current tracking error
        self.e = []  

        # current actor net control action
        self.Ucca = []

        # take control action
        U = Uc + Ucca
    
    def epsilonGreedy(self)
        if P = 1 - epsilon
            Fi = np.tanh(dot(W.transpose(),self.e))
            _Ucca = Fi * V
        else
            _Ucca = Fi * V + np.random.uniform(0.1 * (max(Ucca) - min(Ucca)))


    def trainCriticNet():
        Q_value = Q_value + alpha * (gamma(r - y + _Q_value) - Q_value)

    def FeedForwardNetwork(self, nIn, nHidden, nOut, nLayer)
        # number of neurons in each layer

        self.nIn = nIn
        self.nHidden = nHidden
        self.nOut = nOut
        
        # number of neurons layer
        self.nLayer = nLayer

        # initialize weights randomly(+1 for bias)
        self.hWeights = random.random(self.nHidden, self.nIn+1)
        self.oWeights = random.random((self.nOut, self.nHidden+1))

        # activations of neurons (sum of inputs)
        self.hActivation = zeros((self.nHidden, 1), dtype = float)
        self.oActivation = zeros((self.nOut, 1), dtype = float)

        # outputs of neurons (after sigmoid function)
        self.iOutput = zeros((self.nIn+1, 1), dtype=float)     # +1 for bias
        self.hOutput = zeros((self.nHidden+1, 1), dtype=float) # +1 for bias
        self.oOutput = zeros((self.nOut), dtype=float)

        # deltas for hidden and output layer
        self.hDelta = zeros((self.nHidden), dtype=float)
        self.oDelta = zeros((self.nOut), dtype=float)

    def forward(self, input):
        # set input as output of first layer(bias neuron = 1.0)
        self.iOutput[:-1, 0] = input
        self.iOutput[-1:, 0] = 1.0

        # hidden layer
        self.hActivation = dot(self.hWeights, self.iOutput)
        self.hOutput[-1:, 0] = 1.0

        # hidden layer
        self.hActivation = dot(self.hWeights, self.iOutput)
        self.hOutput[:-1, :] = tanh(self.hActivation)

    def backward(self, teach):      # trained via back propagation(gradient descent), training example provided by critic Net
        error = self.oOutput - array(teach, dtype=float)

        # deltas of output neurons
        self.oDelta = (1 - tanh(self.oActivation)) * tanh(self.oActivation) * error

        # deltas of hidden neurons
        self.hDelta = (1 - tanh(self.hActivation)) * tanh(self.hActivation) * dot(self.oWeights[:, :-1].transpose(), self.oDelta)

        # apply weight changes
        self.hWeights = self.hWeights - self.alpha * dot(self.hDelta, self.iOutput.transpose())
        self.oWeights = self.oWeights - self.alpha * dot(self.oDelta, self.hOutput.transpose())

    def getOutput(self):
        return self.oOutput
        
