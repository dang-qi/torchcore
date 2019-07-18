try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

def convert_state_dict_for_vgg16(state_dict):
    vgg16_keys=['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3']
    origin_key=[0,2,5,7,10,12,14,17,19,21,24,26,28]
    for i, key in enumerate(vgg16_keys):
        state_dict[key+'.weight'] = state_dict['features.{}.weight'.format(origin_key[i])]
        state_dict[key+'.bias'] = state_dict['features.{}.bias'.format(origin_key[i])]
    return state_dict