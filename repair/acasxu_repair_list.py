import torch
from include.utils.vnnlib import vnnlib_to_properties

# the list of neural networks that does not violate any of properties 1-10
safe_nnet_list = [[1,1],[1,2],[1,3],[1,4],[1,5],[1,6], [3,3], [4,2], [1,7], [1,8]]

hard_cases = [[1,9],[2,9]]


# normalize the whole input range to neural networks.
# this is for sampling of training data and test data when they are not available
ranges = torch.tensor([6.02610000e+04, 6.28318531e+00, 6.28318531e+00, 1.10000000e+03,
 1.20000000e+03, 3.73949920e+02])
means = torch.tensor([1.97910910e+04, 0.00000000e+00, 0.00000000e+00, 6.50000000e+02,
 6.00000000e+02, 7.51888402e+00])

#  [Clear-of-Conflict, weak left, weak right, strong left, strong right]
lbs_input = [0.0, -3.141593, -3.141593, 100.0, 0.0]
ubs_input = [60760.0, 3.141593, 3.141593, 1200.0, 1200.0]
for n in range(5):
    lbs_input[n] = (lbs_input[n] - means[n]) / ranges[n]
    ubs_input[n] = (ubs_input[n] - means[n]) / ranges[n]

input_ranges = [lbs_input, ubs_input]

# the list of properties that are violated by at least one neural network
# property7 is only for nnet19, property is only for nnet29
# violated_property = [property1, property2, property3, property4, property7, property8]

# extract properties from vnnlib
vnnlib_path = ''
property1 = vnnlib_to_properties('../nets/prop_1.vnnlib', num_inputs=5, num_outputs=5, input_ranges=input_ranges)[0]
property2 = vnnlib_to_properties('../nets/prop_2.vnnlib', num_inputs=5, num_outputs=5, input_ranges=input_ranges)[0]
property3 = vnnlib_to_properties('../nets/prop_3.vnnlib', num_inputs=5, num_outputs=5, input_ranges=input_ranges)[0]
property4 = vnnlib_to_properties('../nets/prop_4.vnnlib', num_inputs=5, num_outputs=5, input_ranges=input_ranges)[0]
property5 = vnnlib_to_properties('../nets/prop_5.vnnlib', num_inputs=5, num_outputs=5, input_ranges=input_ranges)[0]
property6 = vnnlib_to_properties('../nets/prop_6.vnnlib', num_inputs=5, num_outputs=5, input_ranges=input_ranges)[0]
property7 = vnnlib_to_properties('../nets/prop_7.vnnlib', num_inputs=5, num_outputs=5, input_ranges=input_ranges)[0]
property8 = vnnlib_to_properties('../nets/prop_8.vnnlib', num_inputs=5, num_outputs=5, input_ranges=input_ranges)[0]
property9 = vnnlib_to_properties('../nets/prop_9.vnnlib', num_inputs=5, num_outputs=5, input_ranges=input_ranges)[0]
property10 = vnnlib_to_properties('../nets/prop_10.vnnlib', num_inputs=5, num_outputs=5, input_ranges=input_ranges)[0]

# create neural networks that need to be repaired and their properties
repair_list = []
for i in range(1,6):
    for j in range(1,10):
        nnet = [i,j]
        if nnet in safe_nnet_list or nnet in hard_cases:
            continue
        property_ls =[[property1],
                      [property2],
                      [property3],
                      [property4]]
        repair_list.append([nnet, property_ls])



# nnet [1,9], hard case
property_ls = [[property7]]
repair_list.append([[1,9], property_ls])

#nnet [2,9], hard case
# property_ls= [[property1],
#               [property2],
#               [property3],
#               [property4],
#               [property8]]
property_ls= [[property8]]
repair_list.append([[2, 9], property_ls])


# property_ls =[[property1],
#               [property2],
#               [property3],
#               [property4],
#               [property5],
#               [property6],
#               [property7],
#               [property8],
#               [property9],
#               [property10]]
property_ls= [[property2]]
repair_list.append([[3, 3], property_ls])










