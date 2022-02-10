import torch

def print_mem(device_list=[0], MB=True, msg=None):
    for d in device_list:
        mem = torch.cuda.memory_allocated(d)
        mem_max = torch.cuda.max_memory_allocated(d)
        if MB:
            mem =int(mem / (1024 * 1024))
            mem_max =int(mem_max/ (1024 * 1024))
        last_str='MB' if MB else ''
        #print('current mem in gpu {}: {} {}'.format(d,mem.item(), last_str))
        #print('max mem in gpu {}: {} {}'.format(d,mem_max.item(), last_str))
        if msg is not None:
            print(msg)
        print('mem / max mem {}: {} / {} {}'.format(d,mem, mem_max, last_str))