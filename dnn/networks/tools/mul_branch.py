import torch

def split_dict( inputs, dataset_label, dataset_num=2):
    inds = []
    for i in range(dataset_num):
        ind = torch.where(dataset_label==i)[0]
        inds.append(ind)
    inputs_all = [{} for i in range(dataset_num)]
    for k, v in inputs.items():
        print(k)
        for i,ind in enumerate(inds):
            if isinstance(v, list):
                inputs_all[i][k] = [v[j] for j in ind]
            else:
                inputs_all[i][k] = v[ind]
    return inputs_all

def split_list( the_list, dataset_label, dataset_num=2):
    outputs = []
    for i in range(dataset_num):
        ind = torch.where(dataset_label==i)[0]
        outputs.append([the_list[j] for j in ind])
    return outputs