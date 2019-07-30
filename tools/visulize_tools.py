import matplotlib.pyplot as plt

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

def parse_bench_log(log_path):
    with open(log_path) as f:
        line = f.readline()
        line = line.split()
        epochs = list(range(1, len(line)+1))
        accuracy = [float(x) for x in line]
    return accuracy, epochs

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

def draw_accuracy_all(log_paths, names, out_path, interval=1):
    assert len(names) == len(log_paths)
    f = plt.figure()
    plt.title('Figure for accuracy vs epochs')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    for i, log_path in enumerate(log_paths):
        accuracy, epochs = parse_bench_log(log_path)
        plt.plot(epochs, accuracy, label=names[i])
        plt.scatter(epochs, accuracy)
        if interval > 0:
            for x,y in zip(epochs, accuracy):
                if x%interval==0:
                    plt.annotate(str(y),xy=(x,y))
    plt.legend()

    plt.savefig(out_path, dpi=300)

