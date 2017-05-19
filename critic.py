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
import operator
import scipy.spatial.distance
 
epsilon = 0.9
alpha = 0.1                # learning rate
h = 2                      # network hidden layers
n = 1                      # input vector number
m = 1                      # output vector
W = np.zero(h*n)           # actor network weights,
V = np.zero(m*h)           # actor network weights
r = []                     # reference signal,an external, time varying input signal
gamma = None               # individual hidden unit gain
class CriticNet()
    def __init__(self,)
        # critic weights randomly
        self.qweights = random.random()

        # critic network table to store the value function 
        Q_table = np.zeros(self.e, action)     # each entry in the table refers to the value of a 
                                               # particular state/action pairing

        self.Bp = np.zeros(z)          # allowable actor network perturbations computed in the stability phase
                                      # use them to determine the alowable perturbation ranges R(for each weight in the actor net)
        # the halting criteria
        self.c = np.zeros()           # to prevent an infinite loop situation 
        
        # current tracking error
        self.e = []  

        # current actor net control action
        self.Ucca = []
  
        # take control action      
        self.U = Uc + Ucca               
    # get current actor net control action
    # get the nominal controler vector Uc
    def getNominalControlerVector():
        self.Uc() = getNominalControlerVector()
        return Uc

    def getOverallControlVector(self):
        self.U() = getOverallControlVector()
        U() = Uc() + Ucca()
        return U()

    def getCurrentTrackingError(self):
        self.e = r - Ypo              # r: reference signal, Ypo: plant output
        return self.e
# the epsilon-greedy algorithm uses the actor network output with 
# probability 1 - epsilon or adds a small random perturbation to the 
# actor network output with probability epsilon
    def epsilonGreedy(self, P):       # P: probability
        if P = 1 - epsilon
            Fi = np.tanh(dot(W.transpose(),self.e))
            _Ucca = Fi * V
        else
            _Ucca = Fi * V + np.random.uniform(0.1 * (max(Ucca) - min(Ucca)))
        return _Ucca

    def trainCriticNet():
        if Fi != 0
            gamma = np.tanh(Fi) / Fi     # gamma: individual hidden unit gain
        else gamma = 1
        Q_value = Q_value + alpha * (gamma(r - y + _Q_value) - Q_value)
    
    # there may be a better control signal,Uop which minimizes the sum of
    # future tracking errors. Because the critic network stores the value
    # functions (sum of future tracking errors), we can use the critic to 
    # find the "optimal" control action Uop. Here we define a local 
    # neighborhood around the actor network's current output Uln.
    def getNeighbors(trainingSet, testInstance, k):
        Uln = []          # define a local neighborhood of action Ucca
        length = len(testInstance) - 1
        for x in range(len(trainingSet)):
            dist = np.scipy.spatial.distance.euclidean(testInstance, trainingSet[X], length)
            distances.append((trainingSet[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    # to get the optimum action from the neighborhood of Ucca
    def getOptimumAction(self):
    self.Uop = min(getNeighbors())
    return self.Uop

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
    
    # Update state information
    def Update(self, _e, _Ucca)
        self.e = _e
        self.Ucca = _Ucca

    def getOutput(self):
        return self.oOutput

    def NominalControler(self):
        
