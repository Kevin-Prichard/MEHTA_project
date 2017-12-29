

# RBM implementation
#
# References (in all caps):
# HINTON - "2010 - Hinton - A Practical Guide to Training Restricted Boltzmann Machines"

import numpy as np


# RBM class
#
# - Implements a restricted boltzmann machine using batched constrastive divergence training.
# - Input vectors/training examples/visible units are column vectors of dimension m.
# - Hidden units are an n-dimensional column vector.
# - A set of b training examples are represented as an (m x b) matrix (ie training examples along
#    horz axis)
#
#  input to RBM constructor
#  - num vis units, n_v
#  - num hidden units, n_h
#  - num training examples, m
#  - minibatch size bs
#  - variance of gaussian for initialization of a weights
#
# train method returns:
#  w_ijs - nparray of w_ij for each grad descent step.
#  a, b - values of biases after training
#
# TODO:
#  - add monitor of cost function
#  - implement momentum, RMSProp, Learning rate decay as independent modules
#     that don't add to the complexity of the code (so that turning them on
#     doesn't change form/readibility of code.)
#  - w_ij is currently implemented as (n_v x n_h). Change to (n_h x n_v) matrix to conform to Ng's
#     notation.

class RBMTrans:
    def __init__(self, numVis, numHid, numTrain, bs):
        self.n_v = numVis
        self.n_h = numHid
        self.m = numTrain
        self.batchSize = bs
        self.trained = False

    def setParams(self, W, a, b):
        ''' Sets RBM parameters.
              Inputs: "W" - (n_v x n_h) matrix of trained couplings
                       "a" - (n_v x 1) vector of trained visible biases
                       "b" - (n_h x 1) vector of trained hidden biases
        '''
        self.w_ij, self.a, self.b = W, a, b
        self.trained = True

    def train(self, data, numEpochs, trainRate, biasesTo0=False, allParams=False, print_debug=False,
              l1RegWeight=0, momentum=0, log_interval=10):
        ''' Trains the RBM.
              Inputs:  "data" - (n_v x m) array of inputs
              Returns: "w_ij" - (n_v x n_h) matrix of trained couplings
                       "a" - (n_v x 1) vector of trained visible biases
                       "b" - (n_h x 1) vector of trained hidden biases
        '''
        # make sure data is of specified shape
        assert (data.shape[0] == self.n_v)
        # number of samples must be integer multiple of batchSize
        assert (np.mod(data.shape[1], self.batchSize) == 0)

        batchesInSample = self.m / self.batchSize

        # Init biases
        if biasesTo0:
            self.a, self.b = np.zeros((self.n_v, 1)), np.zeros((self.n_h, 1))
        else:
            # fraction of samples with i'th spin on (HINTON section 8)
            vp = 1. * np.sum(data, axis=1, keepdims=True) / self.m
            self.a, self.b = np.log(vp / (1. - vp)), np.ones((self.n_h, 1))

        # # initialize weights to gaussian small values (HINTON)
        np_rng = np.random.RandomState(1234)
        self.w_ij = np_rng.normal(0, 0.01, size=(self.n_h, self.n_v))

        # For momentum
        v = np.zeros(self.w_ij.shape)

        # For all w_ijs, as, and bs in run
        w_ijs, aa, bb = np.array([self.w_ij]), np.array([self.a]), np.array([self.b])

        # This is the train routine. At the end of this loop, we should have an
        #  array of w_ij matrices -- one for every gradient ascent step.
        if print_debug == True: print("Epoch: (/" + str(numEpochs) + "): ", end="")

        for i in range(numEpochs):
            # randomize sample order for this epoch
            dataThisEpoch = np.copy(data)
            np.random.shuffle(dataThisEpoch.T)

            # make batches for this epoch
            batches = np.split(dataThisEpoch, batchesInSample, axis=1)

            for batch in batches:
                # probability that hidden unit is 1
                # Gives (n_h x batchSize) matrix
                pHidData = self._logistic(np.dot(self.w_ij, batch) + self.b)
                # draw a sample from pHidData
                sampHidData = np.random.binomial(1, pHidData)
                # reconstructed visible pdf from the hidden data sample
                pVisRecon = self._logistic(np.dot(self.w_ij.T, sampHidData) + self.a)
                # sample of this pdf
                sampVisRecon = np.random.binomial(1, pVisRecon)
                # reconstructed hidden pdf
                pHidRecon = self._logistic(np.dot(self.w_ij, pVisRecon) + self.b)
                # <v h> correlations for data and reconstructed
                visHidCorrData = (1. / self.batchSize) * np.dot(pHidData, batch.T)
                visHidCorrRecon = (1. / self.batchSize) * np.dot(pHidRecon, pVisRecon.T)
                # gradient ascent on parameters, with opt L1 regularization
                # TODO check minus sign
                v = momentum * v + trainRate * (visHidCorrData - visHidCorrRecon -
                                                l1RegWeight * np.sign(self.w_ij))
                self.w_ij += v
                # no regularization on biases
                self.a += (trainRate / self.batchSize) * np.sum(batch - pVisRecon, axis=1, keepdims=True)
                self.b += (trainRate / self.batchSize) * np.sum(pHidData - pHidRecon, axis=1, keepdims=True)

            if allParams == True and i % log_interval == 0:
                w_ijs = np.vstack([w_ijs, [self.w_ij]])
                aa = np.vstack([aa, [self.a]])
                bb = np.vstack([bb, [self.b]])

            if print_debug == True and i % log_interval == 0:
                print("%s " % i, end="")

        # kill duplicate first element
        if allParams == True:
            w_ijs, aa, bb = w_ijs[1:], aa[1:], bb[1:]
        else:
            w_ijs, aa, bb = self.w_ij, self.a, self.b

        self.setParams(w_ijs[-1], aa[-1], bb[-1])
        self.trained = True
        print("")
        print("Done")
        return w_ijs, aa, bb

    def vToh(self, vis):
        assert (self.trained == True)
        assert (vis.shape[0] == self.n_v)

        # Calculate final hidden activations from final model
        return self._logistic(np.dot(self.w_ij, vis) + self.b)

    def _logistic(self, x):
        return 1.0 / (1 + np.exp(-x))

    def _tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


  
