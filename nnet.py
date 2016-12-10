from collections import defaultdict
from collections import Counter
from random import random, gauss
#import random
from math import log, e
import pickle
import os

#alpha = 0.0000000000001
alpha = 0.00001
orient = [0,90,180,270]


class Pixel:
    def __init__(self, r, g, b):
        self.r = float(r)
        self.g = float(g)
        self.b = float(b)
        self.intensity = self.find_intensity()

    def find_intensity(self):
        return (0.2989 * self.r) + (0.5870 * self.g) + (0.1140 * self.b)


class Network:
    def __init__(self, input_node_count, hidden_node_count, output_node_count):
        self.input_node_count = int(input_node_count)
        self.hidden_node_count = int(hidden_node_count)
        self.output_node_count = int(output_node_count)
        self.weights = self.assign_random_weights()
        self.L = 3

    def assign_random_weights(self):
        weights = defaultdict(defaultdict)
        hidden_start = self.input_node_count
        hidden_end = hidden_start + self.hidden_node_count
        output_start = hidden_end
        output_end = output_start + self.output_node_count
        for i in range(self.input_node_count):
            for j in range(hidden_start, hidden_end):
                for k in range(output_start, output_end):
                    weights[j][k] = self.get_initial_weight()
                weights[i][j] = self.get_initial_weight()

        return weights

    @staticmethod
    def get_initial_weight():
        return gauss(255/2,255/4)
        #return random()


def get_pixel_objects(pixels):
    l = len(pixels)
    lst = []
    for i in range(0, l, 3):
        temp = pixels[i:i + 3]
        obj = Pixel(temp[0], temp[1], temp[2])
        lst.append(obj)
    return lst


def process_line(line):
    split_line = line.split()
    key = split_line[0]
    orientation = int(split_line[1])
    #pixels = [int(x) for x in split_line[2:]]
    pixels = split_line[2:]
    pixel_obj_lst = get_pixel_objects(pixels)

    return key, orientation, pixel_obj_lst


def get_training_set(filename):
    lst = {}
    with open(filename) as f:
        for line in f:
            key, orientation, pixels = process_line(line)
            key_uniq = ''.join([key,str(orientation)])
            lst[key_uniq] = (pixels, int(orientation))
    return lst


def build_network(image_pixel_count, hidden_node_count, orientation_count):
    return Network(image_pixel_count, hidden_node_count, orientation_count)


def get_orientation_list(orient):
    lst = []
    for x in [0, 90, 180, 270]:
        y = 1 if (orient - x) == 0 else 0
        lst.append(y)
    return lst

#
# def g(x):
#     res = 1 + (e ** x)
#     return log(res)


def g(x):
    return x if x > 0 else (0.01 * x)

# def g_prime(x):
#     res = 1 + (e ** -x)
#     return 1.0 / res

def g_prime(x):
    return 1 if x > 0 else 0.01


def iter_helper(range_start, range_end, array, index, multiplier):
    count = 0
    for j in range(range_start, range_end):
            count += (array[j][index] * multiplier[j])
    return count


def iter_helper_backward(range_start, range_end, array, index, multiplier):
    count = 0
    for j in range(range_start, range_end):
            count += (array[index][j] * multiplier[j])
    return count


def soft_max(input_arr):
    lst = input_arr[:]
    max_ele = max(lst)

    for i in range(len(lst)):
        ele = lst[i]
        lst[i] = ele - max_ele
    sum_val = 0

    for i in range(len(lst)):
        lst[i] = e ** lst[i]
        sum_val += lst[i]

    for i in range(len(lst)):
        lst[i] /= sum_val

    return lst


def scale_outputs(y, out_node_vals):
    if out_node_vals.count(0) == 4:
        return y
    res = []
    max_v = max(out_node_vals)
    for i in range(len(y)):
        x = max_v if y[i] == 1 else 0
        res.append(x)
    return res


def back_prop_learning(examples, network):
    global alpha
    hidden_start = network.input_node_count
    hidden_end = network.input_node_count + network.hidden_node_count
    output_start = hidden_end
    output_end = output_start + network.output_node_count
    for iteration in range(5):

        for example in examples.values():
            y = get_orientation_list(example[1])
            a = defaultdict()
            in_vec = defaultdict()
            x = example[0]
            delta = defaultdict()
            for i in range(network.input_node_count):
                a[i] = x[i].intensity
                in_vec[i] = x[i].intensity

            for j in range(hidden_start, hidden_end):
                # temp = 0
                # for i in range(network.input_node_count):
                #     temp += network.weights[i][j] * a[i]
                in_vec[j] = iter_helper(0, network.input_node_count, network.weights,j, a)
                a[j] = g(in_vec[j])

            out_node_vals = []
            for j in range(output_start, output_end):
                # temp = 0
                # for i in range(hidden_start,hidden_end):
                #     temp += network.weights[i][j] * a[i]
                in_vec[j] = iter_helper(hidden_start, hidden_end, network.weights, j, a)
                a[j] = g(in_vec[j])
                out_node_vals.append(a[j])

            soft_max_vals = soft_max(out_node_vals)
            #expected_vals = scale_outputs(y,out_node_vals)

            for j in range(output_start, output_end):
                delta[j] = g_prime(in_vec[j]) * (y[j - output_start] - soft_max_vals[j - output_start])

            for i in range(hidden_start, hidden_end):
                delta[i] = g_prime(in_vec[i])
                delta[i] *= iter_helper_backward(output_start, output_end, network.weights, i, delta)

            for i in range(network.input_node_count):
                delta[i] = g_prime(in_vec[i])
                delta[i] *= iter_helper_backward(hidden_start, hidden_end, network.weights, i, delta)

            for i in network.weights.keys():
                for j in network.weights[i].keys():
                    network.weights[i][j] += alpha * a[i] * delta[j]
    return network


def serialize_network(neural_network, param):
    f_ptr = open(param, "w")
    pickle.dump(neural_network, f_ptr)
    f_ptr.close()


def deserialize_network(filename):
    f_ptr = open(filename, 'r')
    obj = pickle.load(f_ptr)
    f_ptr.close()
    return obj


def get_orientation_value(index):
    return orient[index]


def test(network, examples):
    hidden_start = network.input_node_count
    hidden_end = network.input_node_count + network.hidden_node_count
    output_start = hidden_end
    output_end = output_start + network.output_node_count
    image_count = len(examples)
    correct_prediction = 0
    predtictions = Counter()

    for example in examples.values():
        true_orientation = example[1]
        a = defaultdict()
        in_vec = defaultdict()
        x = example[0]
        for i in range(network.input_node_count):
            a[i] = x[i].intensity
            in_vec[i] = x[i].intensity

        for j in range(hidden_start, hidden_end):
            # temp = 0
            # for i in range(network.input_node_count):
            #     temp += network.weights[i][j] * a[i]
            in_vec[j] = iter_helper(0, network.input_node_count, network.weights,j, a)
            a[j] = g(in_vec[j])

        out_node_vals = []
        for j in range(output_start, output_end):
            # temp = 0
            # for i in range(hidden_start,hidden_end):
            #     temp += network.weights[i][j] * a[i]
            in_vec[j] = iter_helper(hidden_start, hidden_end, network.weights,j, a)
            a[j] = g(in_vec[j])
            out_node_vals.append(a[j])

        soft_max_vals = soft_max(out_node_vals)

        predicted_orientation = get_orientation_value(soft_max_vals.index(max(soft_max_vals)))

        predtictions[predicted_orientation] += 1

        if true_orientation == predicted_orientation:
            correct_prediction += 1
    print("%d correctly predicted out of %d images" %(correct_prediction,image_count))
    print("Accuracy - %f" %((float(correct_prediction)/image_count) * 100))
    print(predtictions)


def neural_net(train_file, test_file, hidden_node_count,read_cache=False):
    image_pixel_count = 64
    orientation_count = 4
    if hidden_node_count == 0:
        print("Error in input")
        return False

    examples = get_training_set(train_file)
    network = build_network(image_pixel_count, hidden_node_count, orientation_count)
    if os.path.isfile("cache") and read_cache:
        neural_network = deserialize_network("cache")
        print("Loaded from the cache")
    else:
        neural_network = back_prop_learning(examples, network)
        serialize_network(neural_network, "cache")

    print(neural_network)
    print("Training complete")
    examples = get_training_set(test_file)
    res = test(neural_network,examples)
