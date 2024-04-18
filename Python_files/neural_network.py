import numpy as np

class Network:
    def __init__(self):
        self.layers = []
        self.eta = 0.01
        self.lamda = 2
        self.regularizer = 'none'

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)
        
    # predict output for given input
    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.feedforward(output)
        return output
    
    def optimizer(self,eta,lamda,regularizer):
        self.eta = eta
        self.lamda = lamda
        self.regularizer = regularizer
    
    # train the network
    def fit(self, x_train, y_train, x_test, y_test, batch_size, epochs):
        
        no_of_instances = x_train.shape[1]
        no_of_batches = np.ceil(no_of_instances/batch_size).astype('int')
        permuted_indices = np.random.permutation(np.arange(0,no_of_instances))
        x_train = x_train[:,permuted_indices]
        y_train = y_train[:,permuted_indices]
        
        idx = np.arange(0,no_of_instances,batch_size)
        if no_of_batches != len(idx)-1:
            idx = np.append(idx,no_of_instances)
        
        train_loss = np.zeros((epochs,1))
        test_loss = np.zeros((epochs,1))
        # training loop
        for i in range(epochs):
            loss_tr = 0
            for j in range(no_of_batches):
                x = x_train[:,idx[j]:idx[j+1]]
                y = y_train[:,idx[j]:idx[j+1]]
                # forward propagation
                output = x
                for layer in self.layers:
                    output = layer.feedforward(output)

                # compute train loss
                [loss, error]= self.layers[-1].estimate_loss(y, output)
                loss_tr += loss

                for layer in reversed(self.layers):
                    error = layer.backpropogate(error, self.eta, self.lamda, self.regularizer)
            
            #compute test loss
            output = x_test
            for layer in self.layers:
                output = layer.feedforward(output)
                
            [loss_ts, error]= self.layers[-1].estimate_loss(y_test, output)
            
            # calculate average error on all samples
            loss_tr /= no_of_instances
            loss_ts /= no_of_instances
            train_loss[i,0] = loss_tr
            test_loss[i,0] = loss_ts
            print('epoch %d/%d   train_loss=%f   test_loss=%f' % (i+1, epochs, loss_tr, loss_ts), end='\x1b\r')
            
        return train_loss, test_loss
            
            

