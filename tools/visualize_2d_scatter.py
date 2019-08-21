import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.manifold import TSNE

def visualize_2d_scatter(data_path, out_path, dataset, ori_dim, class_num=10, hard_mining=False):
    features = None
    labels = None
    with open(data_path, 'rb') as f:
        data = pickle.load(f) 
        features = data['features']
        labels = data['labels']

    if features is not None and labels is not None:
        #normalize features
        plt.figure()
        if hard_mining:
            plt.title('2D visualization of {}d features for {} dataset with hard mining'.format(ori_dim, dataset))
        else:
            plt.title('2D visualization of {}d features for {} dataset'.format(ori_dim, dataset))
        #features[:,0] = features[:,0] / np.max(np.abs(features[:,0]))
        #features[:,1] = features[:,1] / np.max(np.abs(features[:,1]))
        # use tsne to reduce the dim of features
        dim = features.shape[-1]
        if dim>2:
            features_2d = TSNE(n_components=2).fit_transform(features)
        else:
            features_2d = features
        for i in range(class_num):
            ind = np.where(labels==i)
            feature_i = features_2d[ind]
            feature_x = feature_i[:,0]
            feature_y = feature_i[:,1]
            plt.scatter(feature_x, feature_y, label='{}'.format(i), alpha=0.3)
        plt.legend()
        plt.savefig(out_path, dpi=300)
    else:
        print('Fail to load features and labels from file: {}'.format(data_path))
    