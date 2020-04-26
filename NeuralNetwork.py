import numpy as np

class NeuralNetwork:
    def __init__(self, params):
        self.mountLayer(params)
        self.losses = []
        self.cost = []
        self.accurency = []

    def mountLayer(self, params):
        self.layer={}
        self.activations = params["activation"]
        self.learning_rate = params["learning_rate"]
        for i in range(1, len(params["size"])):
            self.layer["W"+str(i)] = np.array(np.random.rand(params["size"][i], params["size"][i - 1]), dtype=np.float64) * 2 - 1
            self.layer["B"+str(i)] = np.array(np.zeros((params["size"][i],1)), dtype=np.float64)

    def backpropLayer(self, i):
        _DA = self.layer["DA"+str(i)]
        W = self.layer["W"+str(i)]
        Z = self.layer["Z"+str(i)]
        _A = self.layer["A"+str(i - 1)]
        activation =  self.activations[i]
        if (activation == "relu"):
            self.layer["DZ"+str(i)] = self.derivate_relu(_DA, Z)
            self.layer["DW"+str(i)] = np.dot(self.layer["DZ"+str(i)], _A.T)/_A.shape[0]
            self.layer["DB"+str(i)] = np.sum(self.layer["DZ"+str(i)], axis=1, keepdims=True)/_A.shape[0]
            self.layer["DA"+str(i - 1)] = np.dot(W.T, self.layer["DZ"+str(i)])
        elif (activation == "elu"):
            self.layer["DZ"+str(i)] = self.derivate_elu(_DA, Z)
            self.layer["DW"+str(i)] = np.dot(self.layer["DZ"+str(i)], _A.T)/_A.shape[0]
            self.layer["DB"+str(i)] = np.sum(self.layer["DZ"+str(i)], axis=1, keepdims=True)/_A.shape[0]
            self.layer["DA"+str(i - 1)] = np.dot(W.T, self.layer["DZ"+str(i)])
        elif (activation == "sigmoid"):
            self.layer["DZ"+str(i)] = self.derivate_sigmoid(_DA, Z)
            self.layer["DW"+str(i)] = np.dot(self.layer["DZ"+str(i)], _A.T)/_A.shape[0]
            self.layer["DB"+str(i)] = np.sum(self.layer["DZ"+str(i)], axis=1, keepdims=True)/_A.shape[0]
            self.layer["DA"+str(i - 1)] = np.dot(W.T, self.layer["DZ"+str(i)])
        elif (activation == "tanh"):
            self.layer["DZ"+str(i)] = self.derivate_tanh(_DA, Z)
            self.layer["DW"+str(i)] = np.dot(self.layer["DZ"+str(i)], _A.T)/_A.shape[0]
            self.layer["DB"+str(i)] = np.sum(self.layer["DZ"+str(i)], axis=1, keepdims=True)/_A.shape[0]
            self.layer["DA"+str(i - 1)] = np.dot(W.T, self.layer["DZ"+str(i)])
        elif (activation == "swish"):
            self.layer["DZ"+str(i)] = self.derivate_swish(_DA, Z)
            self.layer["DW"+str(i)] = np.dot(self.layer["DZ"+str(i)], _A.T)/_A.shape[0]
            self.layer["DB"+str(i)] = np.sum(self.layer["DZ"+str(i)], axis=1, keepdims=True)/_A.shape[0]
            self.layer["DA"+str(i - 1)] = np.dot(W.T, self.layer["DZ"+str(i)])
        elif (activation == "leaky_relu"):
            self.layer["DZ"+str(i)] = self.derivate_leaky_relu(_DA, Z)
            self.layer["DW"+str(i)] = np.dot(self.layer["DZ"+str(i)], _A.T)/_A.shape[0]
            self.layer["DB"+str(i)] = np.sum(self.layer["DZ"+str(i)], axis=1, keepdims=True)/_A.shape[0]
            self.layer["DA"+str(i - 1)] = np.dot(W.T, self.layer["DZ"+str(i)])
        elif (activation == "mish"):
            self.layer["DZ"+str(i)] = self.derivate_mish(_DA, Z)
            self.layer["DW"+str(i)] = np.dot(self.layer["DZ"+str(i)], _A.T)/_A.shape[0]
            self.layer["DB"+str(i)] = np.sum(self.layer["DZ"+str(i)], axis=1, keepdims=True)/_A.shape[0]
            self.layer["DA"+str(i - 1)] = np.dot(W.T, self.layer["DZ"+str(i)])
    
    def derivate_tanh(self, _DA, x):
        return _DA * (1 - np.tanh(x)**2)

    def derivate_relu(self, _DA, x):
        DZ = np.array(x, copy = True, dtype=np.float64)
        DZ[x < 0] = 0
        return DZ * _DA

    def derivate_sigmoid(self, _DA, x):
        return _DA * (self.sigmoid(x)*(1-self.sigmoid(x)))

    def derivate_leaky_relu(self, _DA, x, alpha=0.01):
        DZ = np.array(x, copy = True, dtype=np.float64)
        DZ[x < 0] = alpha
        return DZ * _DA
        
    def derivate_swish(self, _DA, x, beta = 1):
        return self.swish(x)+_DA*(beta-self.swish(x))
    
    def derivate_elu(self, x, alpha=0.01):
        return np.where(x > 0, 1, self.elu(x, alpha) + alpha)
    
    #def derivate_mish(self, _DA, x):
    #    omega = np.exp(3 * x) + 4 * np.exp(2 * x) + (6 + 4 * x) * np.exp(x) + 4 * (1 + x)
    #    delta = 1 + np.pow((np.exp(x) + 1), 2)
    #    derivative = np.exp(x) * omega / np.pow(delta, 2)
    #    return derivative * _DA
    
    def derivate_mish(self, _DA, x):
        pass
    
    def compute_cost(self, Y, A):
        m = Y.shape[1]
        logprobs = np.multiply(np.log(A), Y) + np.multiply(1 - Y, np.log(1 - A))
        cost = - np.sum(logprobs) / m
        cost = np.squeeze(cost)
        return cost

    def predict(self, input):
        self.layer["A0"] = input
        for i in range(1, len(self.activations)):
            self.forwardLayer(i)
        return self.layer["A"+str(len(self.activations)-1)]

    def train(self, input, output, epoch):
        for _ in range(epoch):
            self.layer["A0"] = input
            for i in range(1, len(self.activations)):
                self.forwardLayer(i)
            self.layer["COST"] = np.subtract(output, self.layer["A"+str(len(self.activations)-1)])
            self.layer["LOSS"] = np.abs(np.subtract(output, self.layer["A"+str(len(self.activations)-1)]))
            self.losses.append(np.sum(self.layer["LOSS"]))
            self.accurency.append(1 - np.mean(self.layer["LOSS"]))
            self.cost.append(np.sum(np.abs(self.layer["COST"])))
            #self.layer["DA" + str(len(self.activations) - 1)] = np.subtract(output, self.layer["A"+str(len(self.activations)-1)]) * self.sigmoid(self.layer["A"+str(len(self.activations)-1)])*(1-self.sigmoid(self.layer["A"+str(len(self.activations)-1)]))
            self.layer["DA"+str(len(self.activations) - 1)] =- (np.divide(output, self.layer["A"+str(len(self.activations)-1)]) - np.divide(1 - output, 1 - self.layer["A"+str(len(self.activations)-1)]))
            for i in range(len(self.activations) - 1, 0, -1):
                self.backpropLayer(i)
            self.learn()

    def learn(self):
        for i in range(1, len(self.activations)):
            self.layer["W"+str(i)] -= self.layer["DW"+str(i)] * self.learning_rate
            self.layer["B"+str(i)] -= self.layer["DB"+str(i)] * self.learning_rate

    def forwardLayer(self, i):
        W = self.layer["W" + str(i)]
        B = self.layer["B" + str(i)]
        _A = self.layer["A"+ str(i - 1)]
        activation = self.activations[i]

        if (activation == "relu"):
            self.layer['Z' + str(i)] = np.dot(W, _A) + B
            self.layer['A' + str(i)] = self.relu(self.layer["Z" + str(i)])
        elif (activation == "elu"):
            self.layer['Z' + str(i)] = np.dot(W, _A) + B
            self.layer['A' + str(i)] = self.elu(self.layer["Z" + str(i)])
        elif (activation == "sigmoid"):
            self.layer['Z' + str(i)] = np.dot(W, _A) + B
            self.layer['A' + str(i)] = self.sigmoid(self.layer["Z" + str(i)])
        elif (activation == "tanh"):
            self.layer['Z' + str(i)] = np.dot(W, _A) + B
            self.layer['A' + str(i)] = np.tanh(self.layer["Z" + str(i)])
        elif (activation == "swish"):
            self.layer['Z' + str(i)] = np.dot(W, _A) + B
            self.layer['A' + str(i)] = self.swish(self.layer["Z" + str(i)])
        elif (activation == "leaky_relu"):
            self.layer['Z' + str(i)] = np.dot(W, _A) + B
            self.layer['A' + str(i)] = self.leaky_relu(self.layer["Z" + str(i)])
        elif (activation == "mish"):
            self.layer['Z' + str(i)] = np.dot(W, _A) + B
            self.layer['A' + str(i)] = self.mish(self.layer["Z" + str(i)])


    def softplus(self, x):
        return np.log(1 + np.exp(x))
    
    def mish(self, x):
        return x * np.tanh(self.softplus(x))

    #def new_mish(self, x):
    #    return x * (np.exp(self.softplus(x)) - np.exp(-self.softplus(x))) / (np.exp(self.softplus(x)) + np.exp(-self.softplus(x)))
    
    def elu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def leaky_relu(self, x):
        return np.where(x > 0, x, x * 0.01)

    def relu(self, x):
        x[x < 0] = 0
        return x
    
    def swish(self, x, beta = 1):
        return ((beta * x) * self.sigmoid(x))
    
    #https://arxiv.org/ftp/arxiv/papers/1908/1908.08681.pdf
    #https://arxiv.org/pdf/1801.07145.pdf