


import scipy.io as sio
import numpy as np

def relu(x):
    return np.maximum(x, 0)

net = sio.loadmat('./nnv_format/ACASXU_run2a_1_1_batch_2000.mat')
print("finished")

# Get the total number of layers in the network
print(net['W'].shape)
print(net['b'].shape)
print(net['act_fcns'].shape)
num_layers = net['W'].shape[1]
print(net['act_fcns'])


#single input
input = np.array([1,2,3,4,5], dtype=np.float32)
# Loop through each layer and print its weights and biases
for i in range(num_layers):
    if(i < num_layers-1):
        weights = net['W'][0, i]
        biases = net['b'][0, i]
        output = relu(np.matmul(weights, input) + np.ravel(biases))
        input = output
    else:
        weights = net['W'][0, i]
        biases = net['b'][0, i]
        output = np.matmul(weights, input) + np.ravel(biases)
        print(output)
       
#multi inputs
input = np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]], dtype=np.float32).T
print(input.shape)
# Loop through each layer and print its weights and biases
for i in range(num_layers):
    if(i < num_layers-1):
        weights = net['W'][0, i]
        print(weights.shape)
        biases = net['b'][0, i]
        print(biases.shape)
        print(input.shape)
        output = relu(np.matmul(weights, input) + biases)
        print("output shape",i, output.shape)
        input = output
    else:
        weights = net['W'][0, i]
        biases = net['b'][0, i]
        output = np.matmul(weights, input) + biases
        print("output shape",i, output.shape)
        print(output)

output = output.T
print(output)