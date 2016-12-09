from sys import argv
import nnet


algo_dict = {
    'nnet': nnet.neural_net
}

if len(argv) < 4:
    print("USAGE: train_file test_file algo [count]")
    exit(1)

train_file = argv[1]
test_file = argv[2]
algo = argv[3]
count = 0
try:
    count = argv[4]
except Exception:
    count = 0

algo_dict[algo](train_file,test_file,count)


