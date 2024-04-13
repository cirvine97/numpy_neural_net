import numpy as np

def relu(Z):
    """Compute the ReLU activation function."""
    return np.maximum(0, Z)


def sigmoid(Z):
    """Compute the sigmoid activation function."""
    return 1 / (1 + np.exp(-1*Z))


def compute_cost(A, Y, type):
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


def dA_final(Y, AL, type):
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


def activation_prime(Z, type):
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


def dA_hidden(W_plus_one, dZ_plus_one):
    """Returns a matrix where each element is the derivative of the cost function wrt each element of a hidden layer activation matrix using the recursion relationship.

    Args:
        W_plus_one (np.array): The weight matrix of the layer one step ahead of this layer.
        dZ_plus_one (np.array): The matrix of derivatives of the cost function wrt linear regressions of the layer one step ahead of this one.

    Returns:
        np.array: the value of dA for this current hidden using the recursion relationship.
    """
    return np.dot(W_plus_one.T, dZ_plus_one)


def intialise_network(
        layer_dims: list
):
    """Initialise the neural network

    Args:
        X (np.array): Matrix containing the training data. Dimension (features, examples)
        layer_dims (list): How many neurons each layer should have. First element must be the number of features in the training data.

    Returns:
        parameters (dict): The initialised weights and bias matrices.
    """
    L = len(layer_dims) - 1
    m = X.shape[1]

    parameters = {}
    for l in range (1, L+1):
        parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.1
        parameters[f"B{l}"] = np.zeros((layer_dims[l], m))

    return parameters


def forwards_pass(
        X: np.array,
        parameters: dict,
        L: int
):
    """Perform a forward pass through the neural network.

    Args:
        X (np.array): Data matrix of dimension (features, examples).
        parameters (dict): Weight and bias matrices throughout the network.
        L (int): How many layers the network has.

    Returns:
        cache (dict): The linear and activation matrices for each layer in the network.
    """
    cache = {}
    cache['A0'] = X

    # All hidden layers use ReLU
    for l in range(1, L+1):
        cache[f"Z{l}"] = np.dot(parameters[f"W{l}"], cache[f"A{l-1}"]) + parameters[f"B{l}"]
        cache[f"A{l}"] = relu(cache[f"Z{l}"])

    # Do final layer with sigmoid 
    cache[f"Z{L}"] = np.dot(parameters[f"W{L}"], cache[f"A{L-1}"]) + parameters[f"B{L}"]
    cache[f"A{L}"] = sigmoid(cache[f"Z{L}"])

    return cache


def backwards_pass(
        Y: np.array,
        parameters: dict,
        cache: dict,
        L: int,
        learning_rate: float,
        type: str
):
        """Perform backpropagation and implement gradient descent on the parameters in all layers within the network.

        Args:
                Y (np.array): Matrix of the training labels.
                parameters (dict): Contains the weights and biases matrices.
                cache (dict): Contains the linear and activation matrices.
                L (int): How many layers does the network have.
                learning_rate (float): How quickly gradient descent should be performed.
                type (str): The type of problem being solved, Can be 'binary classification', 'multilabel classification' or 'regression'.
        
        Returns:
                parameters (dict): Tuned weights and biases matrices after performing gradient descent.
        """
        # Do pass on the final layer first
        if type == 'binary classification':
                dZ = dA_final(Y=Y, AL=cache[f'A{L}'], type=type) * activation_prime(cache[f"Z{L}"], 'sigmoid')
        elif type == 'multilabel classification':
                dZ = dA_final(Y=Y, AL=cache[f'A{L}'], type=type) * activation_prime(cache[f"Z{L}"], 'softmax')
        elif type == 'regression':
                dZ = dA_final(Y=Y, AL=cache[f'A{L}'], type=type)
        else:
                raise ValueError("Choose a valid problem type.")
        
        dW = np.dot(dZ, cache[f"A{L-1}"].T)
        dB = np.sum(dZ, axis=1, keepdims=True)

        # Propagate through the hidden layers
        for l in reversed(range(1, L)):
                # Get dA from information of the front layer
                dA = np.dot( parameters[f"W{l+1}"].T , dZ )

                # Update the parameters of the layer in front of this one via gradient descent now that we no longer need that information
                parameters[f"W{l+1}"] -= learning_rate*dW
                parameters[f"B{l+1}"] -= learning_rate*dB

                dZ = dA * activation_prime(cache[f"Z{l}"] , type='relu')

                dW = np.dot( dZ, cache[f"A{l-1}"].T )
                dB = np.sum( dZ, axis=1, keepdims=True)

        # Update the first layer 
        parameters[f"W{1}"] -= learning_rate*dW
        parameters[f"B{1}"] -= learning_rate*dB

        return parameters


def train_neural_network(
        X: np.array,
        Y: np.array,
        layer_dims: list,
        training_epochs: int,
        learning_rate: float,
        type: str,
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

    costs = []
    parameters = intialise_network(layer_dims)
    L = len(layer_dims) - 1

    for epoch in range(0, training_epochs):
        cache = forwards_pass(X, parameters, L)
        cost = compute_cost(cache[f"A{L}"], Y, type)
        
        # Cost information
        if epoch%100==0 and print_cost:
            print(cost)
        if epoch % 100 == 0 or epoch == training_epochs:
            costs.append(cost)

        parameters = backwards_pass(Y, parameters, cache, L, learning_rate, type)
        
    return parameters, costs
