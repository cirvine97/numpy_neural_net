import numpy as np

class NeuralNet:
    def __init__(
                self, 
                X_train: np.array,
                Y_train: np.array,
                layer_dims: list,
                type: str       
    ):
        """Initialise the neural network with the parameters dictionary as an attribute.

        Args:
            X_train (np.array): Matrix containing the training data. Dimension (features, examples).
            Y_train (np.array): Matrix containing the training labels. Dimension (labels, examples).
            layer_dims (list): How many neurons each layer should have. First element must be the number of features in the training data.
            type (str): The supervised learning task type. Can be one of 'binary classification', 'multilabel classification' or 'regression'.
        """
        self.X = X_train
        self.Y = Y_train
        self.L = len(layer_dims) - 1
        self.m = self.X.shape[1]
        self.type = type

        self.parameters = {}
        for l in range (1, self.L + 1):
            self.parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.1
            self.parameters[f"B{l}"] = np.zeros((layer_dims[l], self.m))


    def relu(
              self, 
              Z: np.array
        ):
        """Compute the ReLU activation function."""
        return np.maximum(0, Z)


    def sigmoid(
              self, 
              Z: np.array
        ):
        """Compute the sigmoid activation function."""
        return 1 / (1 + np.exp(-1*Z))
    

    def softmax(
              self,
              Z: np.array
    ):
        """
        Compute the softmax activation function.
        """
        return np.exp(Z) / float(np.squeeze(np.sum(np.exp(Z), axis=0)))


    def compute_cost(
              self, 
              A, 
              Y, 
              type
        ):
        """Compute the cost of the predictions of the neural network against the training data.

        Args:
            A (np.array): Matrix of the activations of the final layer.
            Y (np.array): Matrix of the true values of the training labels.
            type (str): The supervised learning task type. Can be one of 'binary classification', 'multilabel classification' or 'regression'.

        Returns:
            cost (int): The cost of the network's predictions.
        """
        m = Y.shape[1]

        if type == 'binary classification':
            cost = (-1/m) * np.sum((Y * np.log(A) + (1-Y) * np.log(1-A)), axis=1)
        elif type == 'multilabel classification':
            cost = (-1/m) * np.sum(np.sum((Y * np.log(A)), axis=1), axis=0)
        elif type == 'regression':
            cost = (1/2*m) * np.sum((A - Y)**2, axis=1)
        else:
            return ValueError("Please enter a valid problem type of binary classification, multilabel classification, or regression")

        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        cost = float(cost)

        return cost


    def dA_final(
              self,
              Y, 
              AL, 
              type
        ):
        """Returns a matrix where each element is the derivative of the cost function wrt each element of the final layer activation matrix."""
        m = Y.shape[1]

        if type == 'binary classification':
            dA = -(1/m) * ( np.divide(Y, AL) - np.divide((1-Y), (1-AL)) )
        elif type == 'multilabel classification':
            dA = -(1/m) * np.divide(Y, AL)
        elif type == 'regression':
            dA = -(1/m) * ( Y - AL )
        else:
            raise ValueError("Please enter a valid problem type of binary classification, multilabel classification, or regression")
        
        return dA


    def activation_prime(
              self, 
              Z, 
              type
        ):
        """Return the value of the derivative of the activation function specified at Z."""
        if type == 'sigmoid':
            gprime = np.divide(np.exp(-Z), (1+np.exp(-Z))**2)
        elif type == 'softmax':
            softmax = np.divide(np.exp(Z), np.sum(np.exp(Z), axis=0, keepdims=True))
            gprime = softmax - softmax**2
        elif type == 'relu':
            # Boolean array
            gprime = (Z > 0).astype(int)
        else:
            raise ValueError("Please enter an activation function type of sigmoid, softmax or relu.")
        
        return gprime


    def dA_hidden(
              self,
              W_plus_one, 
              dZ_plus_one
        ):
        """Returns a matrix where each element is the derivative of the cost function wrt each element of a hidden layer activation matrix using the recursion relationship.

        Args:
            W_plus_one (np.array): The weight matrix of the layer one step ahead of this layer.
            dZ_plus_one (np.array): The matrix of derivatives of the cost function wrt linear regressions of the layer one step ahead of this one.

        Returns:
            np.array: the value of dA for this current hidden using the recursion relationship.
        """
        return np.dot(W_plus_one.T, dZ_plus_one)


    def forwards_pass(
              self,
    ):
        """
        Perform a forward pass through the neural network setting the cache of the linear and activation matrices as an attribute.
        """
        self.cache = {}
        self.cache['A0'] = self.X

        # All hidden layers use ReLU
        for l in range(1, self.L+1):
            self.cache[f"Z{l}"] = np.dot(self.parameters[f"W{l}"], self.cache[f"A{l-1}"]) + self.parameters[f"B{l}"]
            self.cache[f"A{l}"] = self.relu(self.cache[f"Z{l}"])

        # Do final layer activation depending on the problem type
        self.cache[f"Z{self.L}"] = np.dot(self.parameters[f"W{self.L}"], self.cache[f"A{self.L-1}"]) + self.parameters[f"B{self.L}"]

        if self.type == 'binary classification':
             self.cache[f"A{self.L}"] = self.sigmoid(self.cache[f"Z{self.L}"])
        elif self.type == 'multilabel classification':
             self.cache[f"A{self.L}"] = self.softmax(self.cache[f"Z{self.L}"])
        elif self.type == 'regression':
             self.cache[f"A{self.L}"] = self.cache[f"Z{self.L}"]
        else:
            raise ValueError("Choose a valid problem type.")


    def backwards_pass(
            self,
            learning_rate: float,
    ):
            """Perform backpropagation and implement gradient descent on the parameters in all layers within the network.

            Args:
                learning_rate (float): How aggressively gradient descent should be performed.
            """
            # Do pass on the final layer first
            if self.type == 'binary classification':
                    dZ = self.dA_final(Y=self.Y, AL=self.cache[f'A{self.L}'], type=self.type) * self.activation_prime(self.cache[f"Z{self.L}"], 'sigmoid')
            elif self.type == 'multilabel classification':
                    dZ = self.dA_final(Y=self.Y, AL=self.cache[f'A{self.L}'], type=self.type) * self.activation_prime(self.cache[f"Z{self.L}"], 'softmax')
            elif self.type == 'regression':
                    dZ = self.dA_final(Y=self.Y, AL=self.cache[f'A{self.L}'], type=self.type)
            else:
                    raise ValueError("Choose a valid problem type.")
            
            dW = np.dot(dZ, self.cache[f"A{self.L - 1}"].T)
            dB = np.sum(dZ, axis=1, keepdims=True)

            # Propagate through the hidden layers
            for l in reversed(range(1, self.L)):
                    # Get dA from information of the front layer
                    dA = np.dot( self.parameters[f"W{l+1}"].T , dZ )

                    # Update the parameters of the layer in front of this one via gradient descent now that we no longer need that information
                    self.parameters[f"W{l+1}"] -= learning_rate*dW
                    self.parameters[f"B{l+1}"] -= learning_rate*dB

                    dZ = dA * self.activation_prime(self.cache[f"Z{l}"] , type='relu')

                    dW = np.dot( dZ, self.cache[f"A{l-1}"].T )
                    dB = np.sum( dZ, axis=1, keepdims=True)

            # Update the first layer 
            self.parameters[f"W{1}"] -= learning_rate*dW
            self.parameters[f"B{1}"] -= learning_rate*dB


    def train_neural_network(
            self,
            training_epochs: int,
            learning_rate: float,
            print_cost: bool = True

    ):
        """Wrapper function to train a fully connected feed-forward neural network.

        Args:
            X (np.array): Data matrix of dimension (features, examples).
            Y (np.array): Matrix of the training labels.
            layer_dims (list): How many neurons each layer should have. First element must be the number of features in the training data.
            training_epochs (int): How many epochs should the network train for.
            learning_rate (float): The learning rate to apply to gradient descent.
            type (str): The type of problem being solved, Can be 'binary classification', 'multilabel classification' or 'regression'.
            print_cost (bool, optional): Whether the cost should be printed every 100 epochs. Defaults to True.

        Returns:
            parameters (dict), costs (list): Weights and bias matrices of the trained network and the costs every 100 epochs.
        """
        # Reproducibility
        np.random.seed(1)
        self.training_epochs = training_epochs
        self.learning_rate = learning_rate

        self.training_costs = []

        for epoch in range(0, training_epochs):
            self.forwards_pass()
            cost = self.compute_cost(self.cache[f"A{self.L}"], self.Y, self.type)
            
            # Cost information
            if epoch%100==0 and print_cost:
                print(f"Epoch {epoch} cost: {cost}")
            if epoch % 100 == 0 or epoch == training_epochs:
                self.training_costs.append(cost)

            self.backwards_pass(self.learning_rate)
