import numpy as np
from activation_functions import activation, derivative

class fcLayer:
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size,act_fn):
        self.weights = np.random.rand(output_size,input_size)*np.sqrt(2/input_size) #he's initialization
        self.bias = np.zeros((output_size,1))
        self.act_fn = act_fn

    # feedword
    def feedforward(self, input_data):
        self.input = input_data
        self.netinput = np.matmul(self.weights,self.input) + self.bias
        self.output = activation(self.netinput,self.act_fn)
        return self.output

    # backpropogation
    def backpropogate(self, output_error, eta, lamda, regularizer):
        n = output_error.shape[1]
        output_error = derivative(self.netinput,self.act_fn)*output_error
        input_error = np.matmul(self.weights.T, output_error)
        errW = np.matmul(output_error,self.input.T)
        errB = np.expand_dims(np.sum(output_error,axis=1)/n,axis=1)
        
        dB = eta*(errB/n)
        if regularizer == 'none':
            dW = eta*(errW/n)
        elif regularizer == 'L1':
            dW = eta*(errW/n) + eta*(lamda/n)*np.sign(self.weights)
        elif regularizer == 'L2':
            dW = eta*(errW/n) + eta*(lamda/n)*self.weights
        
        self.weights -= dW
        self.bias -= dB
        return input_error
    
class classLayer:
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(output_size,input_size)*np.sqrt(1/input_size)
        self.bias = np.zeros((output_size,1))
        
    # predict output for given input
    def estimate_loss(self,y_true,y_pred):
        size  = y_true.shape[0]
        if size == 1:
            loss  = - np.sum(np.log(y_pred)*y_true + np.log(1-y_pred)*(1-y_true))
        else:
            loss  = - np.reduce_sum(np.log(y_pred)*y_true)
        error = y_pred-true
        return (loss, error)

    # feedword
    def feedforward(self, input_data):
        self.input = input_data
        self.netinput = np.matmul(self.input, self.weights) + self.bias
        self.output = np.exp(self.netinput)
        self.output = self.output/np.sum(self.output,axis=0)
        return self.output

    # backpropogation
    def backpropogate(self, output_error, eta, lamda, regularizer):
        n = output_error.shape[1]
        input_error = np.matmul(self.weights.T, output_error)
        errW = np.matmul(output_error,self.input.T)
        errB = np.expand_dims(np.sum(output_error,axis=1)/n,axis=1)

        # update parameters
        dB = eta*(errB/n)
        if regularizer == 'none':
            dW = eta*(errW/n)
        elif regularizer == 'L1':
            dW = eta*(errW/n) + eta*(lamda/n)*np.sign(self.weights)
        elif regularizer == 'L2':
            dW = eta*(errW/n) + eta*(lamda/n)*self.weights
        
        self.weights -= dW
        self.bias -= dB
        return input_error
    
class regLayer(fcLayer):
    def estimate_loss(self,y_true,y_pred):
        loss  = 0.5*np.sum(np.mean((y_pred-y_true)**2,axis=0));
        error = y_pred-y_true
        return (loss, error)

