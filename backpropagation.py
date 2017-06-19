import numppy as np
from numpy import *

class Bp:
    def __init__(nInput, nHidden, nOutput):
        #learning rate
        self.alpha = 0.1

        self.nInput = nInput 
        self.nHidden = nHidden
        self.nOutput = nOutput

        self.ihWeight = np.random.randn((self.nInput + 1,self.nHidden)) / sqrt(self.nInput)
        self.hoWeight = np.random.randn((self,nHidden +1, self.nOutput)) / sqrt(self.nInput)
        self.hThreshold = np.random.random((self.nInput,1))
        self.oThreshold = np.random.random((self.nOutput,1))
        
        self.ihWeightSum = np.zeros((1, self.nHidden))
        self.hoWeightSum = np.zeros((1, self.nOutput))
        
        self.oIncrementWeight = np.zeros((self.nOutput, self.nHidden))
        self.hIncrementWeight = np.zeros((self.nHidden, self.nInput))

    def FeedForward(self):
        # calculate the output of hidden layer units
        self.ihWeightSum = self.iOutput.dot(self.ihWeight)
        self.z1 = self.ihWeightSum - self.hThreshold 
        self.hOutput = np.tanh(self.z1)
    
        # caluculate the output of output layer unit
        self.hoWeightSum = self.hOutput.dot(self.hoWeight)
        self.z2 = self.hoWeightSum - self.oThreshold
        self.oOutput = np.tanh(self.z2)

    def BackPropagate(self):
        #calculate the error in output layer
        for j in range(1:self.nOutput)
            self.oError[j] = (self.oOutput[j] - self.Yr[j])**2
            self.ErrorSum += self.oError[j]
        self.mse = 1/2 * self.ErrorSum    # calculate the mean square error
        
        #calculate the delta value in output layer
        self.oDelta = (1 - self.oOutput**2)

        #calculate the delta value in Hidden layer
        for i range(1:self.nHidden)
            for j range(1:self.nOutput)
                self.w += self.oDelta[j]*self.hoWeight[i][j]
            self.hDelta[i] = self.w.dot(1 - self.oOutput**2)
        
        # calculate the increment Weight between Output layer and Hidden layer
        for j in range(1:self.nOutput)
            for i in range(1:self.nHidden)
                self.oIncrementWeight[j][i] = -(1 - self.oOUtput[j]**2) * (self.Yr[j] - self.oOutput[j]) * self.hOutput[i]
    
        #calculate the increment Weight between Hidden layer and Input layer 
        for i in range(1:nHidden)
            self.hIncrementWeight = -alpha * self.hDelta.dot(self.oOutput)

       
        # update the weight in total network
        self.hoWeight = self.hoWeight + self.oIncrementWeight
        self.ihweight = self.ihweight + self.hIncrementWeight
        
        
        
