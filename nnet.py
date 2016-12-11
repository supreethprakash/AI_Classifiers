import os
import pickle
from collections import Counter
from collections import defaultdict
from math import e
from random import gauss, sample


alpha = 0.00001
orient = [0, 90, 180, 270]


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
        # self.hidden_start = self.input_node_count
        self.hidden_end = self.input_node_count + self.hidden_node_count
        # self.output_start = self.hidden_end
        self.output_end = self.hidden_end + self.output_node_count
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
        # return gauss(255/2,255/4)
        # return gauss(0.25, 0.1)
        return gauss(0.001,0.01)


class Propagator:
    def __init__(self, ground_truth):
        self.y = get_orientation_list(ground_truth)
        self.a = defaultdict()
        self.in_vec = defaultdict()
        self.delta = defaultdict()
        self.soft_max_vals = []


class Result:
    """
    Has the confusion matrix, images classified correctly and total number of images tested against.
    Also handles displaying the results on the console.
    Taken from my Assignment 4 implementation.
    """

    def __init__(self, confusion_matrix, correct_classification, total_documents):
        self.confusion_matrix = confusion_matrix
        self.right = correct_classification
        self.total = total_documents

    def __str__(self):

        str_formatter = ["\t"] + ["    ".join(self.confusion_matrix.keys())]
        str_formatter.append("\n")
        for key in self.confusion_matrix.keys():
            str_formatter.append(key +" " * (3 - len(key)))
            for res in self.confusion_matrix.keys():
                str_formatter.append(str(self.confusion_matrix[key][res]))
                str_formatter.append("    ")
            str_formatter.append("\n")
        str_formatter.append("\nCorrect Classification: %d\n" % self.right)
        str_formatter.append("\nTotal number of images: %d\n" % self.total)
        str_formatter.append("\nAccuracy: %f\n" % round(((float(self.right) / self.total) * 100), 3))

        return ' '.join(str_formatter)


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
    # pixels = [int(x) for x in split_line[2:]]
    pixels = split_line[2:]
    pixel_obj_lst = get_pixel_objects(pixels)

    return key, orientation, pixel_obj_lst


def get_training_set(filename):
    lst = {}
    with open(filename) as f:
        for line in f:
            key, orientation, pixels = process_line(line)
            key_uniq = ''.join([key, str(orientation)])
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


def propagate_forward(network, x, ground_truth):
    propagator = Propagator(ground_truth)
    a = propagator.a
    in_vec = propagator.in_vec

    for i in range(network.input_node_count):
        a[i] = x[i].intensity
        in_vec[i] = x[i].intensity

    for j in range(network.input_node_count, network.hidden_end):
        in_vec[j] = iter_helper(0, network.input_node_count, network.weights, j, a)
        a[j] = g(in_vec[j])

    out_node_vals = []
    for j in range(network.hidden_end, network.output_end):
        in_vec[j] = iter_helper(network.input_node_count, network.hidden_end, network.weights, j, a)
        a[j] = g(in_vec[j])
        out_node_vals.append(a[j])

    soft_max_vals = soft_max(out_node_vals)

    propagator.soft_max_vals = soft_max_vals[:]

    return propagator


def propagate_backward(network, propagator):
    for j in range(network.hidden_end, network.output_end):
        propagator.delta[j] = g_prime(propagator.in_vec[j]) * (
        propagator.y[j - network.hidden_end] - propagator.soft_max_vals[j - network.hidden_end])

    for i in range(network.input_node_count, network.hidden_end):
        propagator.delta[i] = g_prime(propagator.in_vec[i])
        propagator.delta[i] *= iter_helper_backward(network.hidden_end, network.output_end, network.weights, i,
                                                    propagator.delta)

    for i in range(network.input_node_count):
        propagator.delta[i] = g_prime(propagator.in_vec[i])
        propagator.delta[i] *= iter_helper_backward(network.input_node_count, network.hidden_end, network.weights, i,
                                                    propagator.delta)

    for i in network.weights.keys():
        for j in network.weights[i].keys():
            network.weights[i][j] += alpha * propagator.a[i] * propagator.delta[j]

    return network, propagator


def learn(examples, network, linear_iter_count=1):
    global alpha
    alpha = 1.0 / len(examples)
    print("Learning rate: %f" % alpha)

    local_minima_count = 0
    iteration = 0
    sample_size = 200
    iteration_count = len(examples) / sample_size
    satisfactory_prediction_count = 0.85 * sample_size

    for i in range(linear_iter_count):
        for example in examples.values():
            ground_truth = int(example[1])
            x = example[0]

            propagator = propagate_forward(network, x, ground_truth)
            network, propagator = propagate_backward(network, propagator)

            del propagator

    print("Linear propagation complete.")

    while local_minima_count < 1:# and iteration < 2000:
        iteration += 1
        right_predictions = 0
        sample_values = sample(examples.values(), sample_size)
        for example in sample_values:
            x = example[0]
            propagator = propagate_forward(network, x, int(example[1]))
            soft_max_vals = propagator.soft_max_vals
            predicted_orientation = get_orientation_value(soft_max_vals.index(max(soft_max_vals)))
            if int(example[1]) == predicted_orientation:
                right_predictions += 1
            network, propagator = propagate_backward(network, propagator)

            del propagator

        print("Right Predictions - %d/%d" % (right_predictions, sample_size))

        if right_predictions >= satisfactory_prediction_count:
            print("Hitting Local minima.")
            local_minima_count += 1
            iteration = 1
            # satisfactory_prediction_count += 0.01 * sample_size
            # break

        if iteration >= iteration_count:
            satisfactory_prediction_count -= 0.01 * sample_size
            print("Expectations are now %d" % satisfactory_prediction_count)
            iteration = 1

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
    correct_prediction = 0
    predictions = Counter()
    image_count = len(examples)
    confusion_matrix = defaultdict(Counter)

    for example in examples.values():
        true_orientation = example[1]
        x = example[0]

        propagator = propagate_forward(network, x, true_orientation)

        soft_max_vals = propagator.soft_max_vals

        predicted_orientation = get_orientation_value(soft_max_vals.index(max(soft_max_vals)))

        predictions[predicted_orientation] += 1
        confusion_matrix[str(true_orientation)][str(predicted_orientation)] += 1

        if true_orientation == predicted_orientation:
            correct_prediction += 1

    result = Result(confusion_matrix, correct_prediction, image_count)
    return result


def neural_net(train_file, test_file, hidden_node_count, cache_file=False):
    print("Technique being used: Neural net.")
    image_pixel_count = 64
    orientation_count = 4
    if hidden_node_count == 0:
        print("Error in input")
        return False

    if cache_file and os.path.isfile(cache_file):
        neural_network = deserialize_network(cache_file)
        print("Loaded the neural net from cache. That was easy.")
    else:
        examples = get_training_set(train_file)
        print("Training set loaded.")
        network = build_network(image_pixel_count, hidden_node_count, orientation_count)
        print("Initial Network built.")
        neural_network = learn(examples, network,0)
        if cache_file:
            serialize_network(neural_network, cache_file)

    print(neural_network)
    print("Training complete")
    examples = get_training_set(test_file)
    print("Loaded Test files...")

    res = test(neural_network, examples)

    print(res)
