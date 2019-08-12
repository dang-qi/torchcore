import matplotlib.pyplot as plt
import numpy as np

def parse_loss_log(log_path, losses_names):
    epoch = 0
    pre_iter = 0
    iters = []
    losses = {}
    pre_total_iter = 0
    for name in losses_names:
        losses[name]=[]
    
    with open(log_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            if line[0]=='epoch':
                epoch=int(line[1])
                pre_total_iter = pre_total_iter + pre_iter
            else:
                iterations = int(line[0]) + pre_total_iter
                iters.append(iterations)
                assert len(losses_names) == len(line)-1
                for i in range(1, len(line)):
                    losses[losses_names[i-1]].append(float(line[i]))
                pre_iter = int(line[0])
    return losses, iters

def draw_loss(log_path, out_path, losses_names=['loss']):
    f = plt.figure()
    losses, iters = parse_loss_log(log_path, losses_names)
    plt.title('Figure for loss vs iterations')
    for name, loss in losses.items():
        plt.plot(iters, loss, label=name)
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.savefig(out_path)

def parse_bench_log(log_path, type_num=1):
    with open(log_path) as f:
        line = f.readline()
        line = line.split()
        accuracy = np.array([float(x) for x in line]).reshape(-1, type_num)
        accuracies = []
        for i in range(type_num):
            accuracies.append(accuracy[:, i])
        epochs = list(range(1, len(accuracy)+1))
    return accuracies, epochs

def draw_accuracy(log_path, out_path, interval=1):
    f = plt.figure()
    accuracy, epochs = parse_bench_log(log_path)
    plt.title('Figure for accuracy vs epochs')
    plt.plot(epochs, accuracy, color='#B16FDE')
    plt.scatter(epochs, accuracy, color='#551A7C')
    for x,y in zip(epochs, accuracy):
        if x%interval==0:
            plt.annotate(str(y),xy=(x,y))
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.savefig(out_path)

def draw_accuracy_all(log_paths, names, out_path, dataset_name, interval=1, type_num=1, type_names=['']):
    assert len(names) == len(log_paths)
    assert type_num == len(type_names)
    for i in range(type_num):
        f = plt.figure(i+1)
        plt.title('{}--Figure for {} accuracy vs epochs'.format(dataset_name, type_names[i]))
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        for j, log_path in enumerate(log_paths):
            accuracies, epochs = parse_bench_log(log_path, type_num=type_num)
            accuracy = accuracies[i]
            plt.plot(epochs, accuracy, label=names[j])
            plt.scatter(epochs, accuracy)
            if interval > 0:
                for x,y in zip(epochs, accuracy):
                    if x%interval==0:
                        plt.annotate(str(y),xy=(x,y))
        plt.legend()

        plt.savefig(out_path.format(type_names[i]), dpi=300)

def draw_pair_class_accuracy(log_path, name, out_path, dataset_name, interval=0, type_num=1, type_names=[], ignore_names=[]):
    assert type_num == len(type_names)
    f=plt.figure()
    plt.title('{}--Figure for {} vs epochs'.format(dataset_name, name))
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    for i in range(type_num):
        if type_names[i] in ignore_names:
            continue
        accuracies, epochs = parse_bench_log(log_path, type_num=type_num)
        accuracy = accuracies[i]
        plt.plot(epochs, accuracy, label=type_names[i])
        plt.scatter(epochs, accuracy)
        if interval > 0:
            for x,y in zip(epochs, accuracy):
                if x%interval==0:
                    plt.annotate(str(y),xy=(x,y))
    plt.legend()

    plt.savefig(out_path.format('pair_classification'), dpi=300)

