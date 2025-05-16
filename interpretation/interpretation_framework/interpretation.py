import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import torch
import captum
from captum.attr import IntegratedGradients, InputXGradient, GuidedBackprop, Deconvolution
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
algorithm_map = {
    'InputXGradient': InputXGradient,
    'Deconvolution': Deconvolution,
    'GuidedBackprop': GuidedBackprop,
    'IntegratedGradients': IntegratedGradients
}

class TPForwardWrapper(torch.nn.Module):
    def __init__(self, model, indices):
        super().__init__()
        self.model = model
        self.indices = indices

    def forward(self, x):
        return self.model(x)[self.indices, :]

def cnn_interpretation_pipeline(model, loader_train, loader_test, width, features_num, save_filename, 
                                algorithm='IntegratedGradients', filter_type = "postfiltering", need_return=1):

    mean_train, cnt_train = process_dataset(model, loader_train, width, features_num, algorithm, filter_type)
    mean_test, cnt_test = process_dataset(model, loader_test, width, features_num, algorithm, filter_type)

    print('done interpretation')

    mean = (mean_train + mean_test) / (cnt_train + cnt_test)
    mean = torch.from_numpy(mean)
    print(f'Averaged tensor shape: {mean.shape}')
    print(f'Averaged tensor: {mean}')

    torch.save(mean, f'{save_filename}.pt')

    if need_return:
      return mean
    else:
      return

def get_ranked_features(features_weights):
    features = features_weights
    p_deviation = pd.DataFrame()

    for column in features_weights.columns:
        if column == 'Unnamed: 0':
            continue

        mean = features_weights[column].mean()
        p_deviation[f'{column}_p_deviation'] = (((features_weights[column] - mean) / mean) * 100).abs()

    p_deviation['mean_deviation'] = p_deviation.mean(axis=1)
    features_range = p_deviation[['mean_deviation']].sort_values(by='mean_deviation', ascending=False)

    return features_range

def process_dataset(model, dataset, width, features_num, 
                    algorithm="IntegratedGradients", filter_type = "postfiltering"):
    mean = np.zeros(features_num + 4, dtype=float)
    cnt = 0
    
    for x, y_true in dataset:
        x, y_true = x.to(device), y_true.to(device).long()
        output = model(x)
        y_pred = torch.argmax(output, dim=1).reshape(1, width)

        if filter_type == "prefiltering":
            explanation = interpretation_prefiltering(model, x, y_true, y_pred, algorithm)
        else:
            explanation = interpretation_postfiltering(model, x, y_true, y_pred, width, algorithm)
        
        if not np.isnan(explanation).any():
            mean += explanation
            cnt += 1
            
    return mean, cnt


# for gradient methods only
def interpretation_prefiltering(model, x, y_true, y_pred, algorithm="IntegratedGradients"):
    tp_mask = (y_true == 1) & (y_pred == 1)
    tp_indices = torch.where(tp_mask)[0]

    if len(tp_indices) == 0:
        return np.array([float('nan')]) 

    wrapper = TPForwardWrapper(model, tp_indices)
    explain = algorithm_map.get(algorithm)(wrapper)
    
    if algorithm == "IntegratedGradients":
        explanation = explain.attribute(x, target=1, n_steps=1)
    else:
        explanation = explain.attribute(x, target=1)

    return explanation.squeeze(dim=0).mean(dim=0).cpu().detach().numpy()

def interpretation_postfiltering(model, x, y_true, y_pred, width, algorithm="IntegratedGradients"):
    idxs = []
    for i in range(width):
        if y_pred[0][i] == y_true[0][i] and y_true[0][i] == 1:
            idxs.append(i)

    explain = algorithm_map.get(algorithm)(model)
    
    if algorithm =='IntegratedGradients':
        explanation = explain.attribute(x, target=1, n_steps=1)
    else:
        explanation = explain.attribute(x, target=1)
    explanation = explanation.squeeze(dim=0)
    
    if len(idxs) > 0:
        return explanation[idxs, :].mean(dim=0).cpu().detach().numpy()
    else:
        return np.array([float('nan')])