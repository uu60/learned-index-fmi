import json
import math
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import pandas as pd
import random

from collections import OrderedDict

# CASE responds to the dataset name, such as lognormal5 and trans
CASE = 'None'

def normalize_data(data, mean=None, std=None):
    if mean is None:
        mean = np.mean(data)
        std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data, mean, std

def denormalize_data(data, mean, std):
    return (data * std) + mean

# Get the index column name for each dataset
def get_column_name():
    if CASE.startswith("trans"):
        return "account_id"
    elif CASE.startswith("bureau"):
        return "SK_ID_BUREAU"
    else:
        return 'key'

def create_model(net_config, num_experts, dropout_prob, device):
    class DualLayerNet(nn.Module):
        def __init__(self, config, num_experts, dropout_prob):
            super(DualLayerNet, self).__init__()
            self.num_experts = num_experts
            input_size = 1

            self.gating_network = None

            self.expert_networks = nn.ModuleList()
            for _ in range(num_experts + 1 if num_experts > 1 else num_experts):
                layers = []
                layers.append(nn.Linear(input_size, config['0']))
                layers.append(nn.ReLU())
                if dropout_prob > 0:
                    layers.append(nn.Dropout(dropout_prob))

                for i in range(1, len(config)):
                    layers.append(nn.Linear(config[str(i - 1)], config[str(i)]))
                    layers.append(nn.ReLU())
                    if dropout_prob > 0:
                        layers.append(nn.Dropout(dropout_prob))

                if self.num_experts > 1 and _ == 0:
                    layers.append(nn.Linear(config[str(len(config) - 1)], num_experts))
                    self.gating_network = nn.Sequential(*layers)
                else:
                    layers.append(nn.Linear(config[str(len(config) - 1)], 1))
                    expert_model = nn.Sequential(*layers)
                    self.expert_networks.append(expert_model)

        def forward(self, x):
            if self.num_experts > 1:
                gating_logits = self.gating_network(x)  # (batch_size, expert_num)
                gating_weights = F.softmax(gating_logits, dim=1)  # (batch_size, expert_num)

                expert_outputs = []
                for expert in self.expert_networks:
                    expert_output = expert(x)  # (batch_size, 1)
                    expert_outputs.append(expert_output)

                expert_outputs = torch.stack(expert_outputs, dim=2)  # (batch_size, 1, expert_num)

                gating_weights = gating_weights.unsqueeze(2)  # (batch_size, expert_num, 1)

                final_output = torch.bmm(expert_outputs, gating_weights).squeeze(2)  # (batch_size, 1)
            else:
                final_output = self.expert_networks[0](x)

            return final_output

    def replace_batchnorm_with_groupnorm(model):
        for child_name, child in model.named_children():
            if isinstance(child, nn.BatchNorm1d):
                setattr(model, child_name, nn.GroupNorm(1, child.num_features))
            elif isinstance(child, nn.BatchNorm2d):
                setattr(model, child_name, nn.GroupNorm(1, child.num_features))
            else:
                replace_batchnorm_with_groupnorm(child)
        return model
    model = DualLayerNet(net_config, num_experts, dropout_prob).to(device)
    model = replace_batchnorm_with_groupnorm(model)
    return model

# There are dummy prefix to the weights, which can be removed by this function.
def handle_weights(weights):
    new_state_dict = OrderedDict()
    for k, v in weights.items():
        if k.startswith("_module."):
            new_key = k.replace("_module.", "")
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict

def get_all_unique_keys():
    file_path = f'dataset/{CASE}.csv'
    df = pd.read_csv(file_path)
    return np.sort(np.unique(df[get_column_name()].to_numpy()))

noise = {}

def get_model_max_error():
    global noise
    if CASE not in noise:
        file_path = f'dataset/{CASE}.csv'
        df = pd.read_csv(file_path)
        noise[CASE] = np.random.exponential(scale=(len(df) / 1000))

    n = noise[CASE]

    # train records of 250k version of bureau do not be recorded
    # 1067.9844 is shown in the training log output
    if CASE == "bureau_sampled_250k":
        return math.ceil(1067.9844 + n)
    model_log_path = f'record/training_results_{CASE.replace("_sampled", "")}.csv'
    df = pd.read_csv(model_log_path)
    min_loss_row = df.loc[df['average_validation_loss'].idxmin()]
    max_error_at_min_loss = min_loss_row['max_error']
    return math.ceil(max_error_at_min_loss + n)

def get_noise_overheads(idxes, relative=False):
    pairs = []
    for idx in idxes:
        pairs.append((idx, idx))
    return get_noise_overheads_for_intervals(pairs, relative)

def get_noise_overheads_for_intervals(idx_pairs, relative=False):
    data = pd.read_csv(f'dataset/{CASE}.csv')[get_column_name()].to_numpy()
    sorted_data = np.sort(data)
    keys, labels = np.unique(sorted_data, return_index=True)

    noise_file = f'noise/{CASE}.json'
    if not os.path.exists(noise_file):
        raise FileNotFoundError(f"Noise index file {noise_file} does not exist")

    with open(noise_file, 'r') as f:
        noise_data = json.load(f)

    overheads = []
    indexes = noise_data.get("indexes", [])
    for idx_pair in idx_pairs:
        low_value = indexes[idx_pair[0]][0]
        high_value = indexes[idx_pair[1] + 1][1]

        overhead = high_value - low_value - (labels[idx_pair[1] + 1] - labels[idx_pair[0]])
        if relative:
            overhead = overhead / (labels[idx_pair[1] + 1] - labels[idx_pair[0]])
        overheads.append(overhead)

    return overheads

def get_single_noise_missing_data(idxes):
    pairs = []
    for idx in idxes:
        pairs.append((idx, idx))
    return get_single_noise_missing_data_for_intervals(pairs)

def get_single_noise_missing_data_for_intervals(idx_pairs):
    noise_file = f'crypt/{CASE}.json'
    if not os.path.exists(noise_file):
        raise FileNotFoundError(f"Noise index file {noise_file} does not exist")

    with open(noise_file, 'r') as f:
        noise_data = json.load(f)

    data = pd.read_csv(f'dataset/{CASE}.csv')[get_column_name()].to_numpy()
    sorted_data = np.sort(data)
    keys, labels = np.unique(sorted_data, return_index=True)

    missing_datas = []
    indexes = noise_data.get("indexes", [])
    for idx_pair in idx_pairs:
        low_value = indexes[idx_pair[0]][0]
        high_value = indexes[idx_pair[1]][1] - 1

        max_real = labels[idx_pair[1] + 1] - 1
        min_real = labels[idx_pair[0]]

        overlap_start = max(min_real, low_value)
        overlap_end = min(max_real, high_value)

        real_range = max_real - min_real + 1
        overlap_range = max(0, overlap_end - overlap_start + 1)

        missing_data = (real_range - overlap_range) / real_range
        missing_datas.append(missing_data)

    return missing_datas

def get_model_overheads(idxes, keys, device, relative=False):
    pairs = []
    for idx in idxes:
        pairs.append((idx, idx))
    return get_model_overheads_for_intervals(pairs, keys, device, relative)

def get_model_overheads_for_intervals(idx_pairs, keys, device, relative=False):
    # prepare inputs for the model
    data = np.sort(pd.read_csv(f'dataset/{CASE}.csv')[get_column_name()].to_numpy())
    keys_normalized, mean_keys, std_keys = normalize_data(data)
    _, labels = np.unique(keys_normalized, return_index=True)
    _, mean_labels, std_labels = normalize_data(labels)

    model = create_model(NET_CONFIG, NUM_EXPERTS, DROPOUT_PROB, device)
    weights = handle_weights(torch.load(f'weights/best-{CASE}.pth'.replace("_sampled", ""), weights_only=True))
    model.load_state_dict(weights)
    model.eval()
    model.to(device)

    overheads = []

    for idx_pair in idx_pairs:
        key = keys[idx_pair[0]]
        next_key = keys[idx_pair[1] + 1]
        inputs = torch.tensor([[normalize_data(key, mean_keys, std_keys)[0]], [normalize_data(next_key, mean_keys, std_keys)[0]]], dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(inputs)

        preds = denormalize_data(outputs, mean_labels, std_labels).cpu().tolist()
        pred0 = math.ceil(max(preds[0][0], 0))
        pred1 = math.floor(max(preds[1][0], 0))
        max_error = get_model_max_error()
        overhead = pred1 - pred0 + 1 + 2 * max_error - (labels[idx_pair[1] + 1] - labels[idx_pair[0]])

        if relative:
            overhead = overhead / (labels[idx_pair[1] + 1] - labels[idx_pair[0]])
        overheads.append(overhead)

    return overheads

# Get random points to evaluate point query overhead
def get_random_point_idxes(length):
    compare_num = min(length, 100)
    if compare_num == 0 or length == 0:
        return []
    step = length // compare_num

    selected_idxes = []
    for i in range(compare_num):
        start = i * step
        end = min((i + 1) * step, length - 1)  # 确保不超出 keys 长度
        if start < end:
            selected_idxes.append(random.randint(start, end - 1))

    return selected_idxes

# Get random intervals to evaluate range query overhead (min length of the interval is 2)
def get_random_interval_idxes(length):
    num_interval = min(50, length // 3)
    interval_length = max(2, math.ceil(length * 0.01))  # 至少为1，防止过小数据集
    step = length // num_interval

    intervals = []
    for i in range(num_interval):
        start = i * step
        end = min((i + 1) * step, length - interval_length - 2)
        start_idx = random.randint(start, end)
        end_idx = start_idx + interval_length
        intervals.append((start_idx, end_idx))

    return intervals

# the main process
def evaluate_overhead():
    global CASE, NUM_EXPERTS, NET_CONFIG, DROPOUT_PROB
    cases = ['uniform3', 'uniform4', 'uniform5', 'uniform6',
             'lognormal3', 'lognormal4', 'lognormal5', 'lognormal6',
             'trans_sampled',
             'bureau_sampled_250k', 'bureau_sampled']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Available GPU: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU available")
    for relative in [True, False]:
        point_results = {}
        interval_results = {}
        for c in cases:
            CASE = c
            if CASE == 'lognormal3':
                NUM_EXPERTS = 2
                NET_CONFIG = {'0': 24, '1': 12, '2': 6}
                DROPOUT_PROB = 0.0
            elif CASE == 'lognormal4':
                NUM_EXPERTS = 2
                NET_CONFIG = {'0': 96, '1': 64, '2': 32}
                DROPOUT_PROB = 0.0
            elif CASE == 'lognormal5':
                NUM_EXPERTS = 2
                NET_CONFIG = {'0': 320, '1': 160, '2': 80}
                DROPOUT_PROB = 0.0
            elif CASE == 'lognormal6':
                NUM_EXPERTS = 2
                NET_CONFIG = {'0': 320, '1': 160, '2': 80}
                DROPOUT_PROB = 0.0
            elif CASE == 'uniform3':
                NUM_EXPERTS = 1
                NET_CONFIG = {'0': 32, '1': 24, '2': 16}
                DROPOUT_PROB = 0.0
            elif CASE == 'uniform4':
                NUM_EXPERTS = 1
                NET_CONFIG = {'0': 32, '1': 24, '2': 16}
                DROPOUT_PROB = 0.0
            elif CASE == 'uniform5':
                NUM_EXPERTS = 1
                NET_CONFIG = {'0': 48, '1': 36, '2': 24}
                DROPOUT_PROB = 0.0
            elif CASE == 'uniform6':
                NUM_EXPERTS = 1
                NET_CONFIG = {'0': 48, '1': 40, '2': 32}
                DROPOUT_PROB = 0.0
            elif CASE == 'trans_sampled':
                NUM_EXPERTS = 2
                NET_CONFIG = {'0': 500, '1': 550, '2': 500}
                DROPOUT_PROB = 0.0
            elif CASE == 'bureau_sampled_250k':
                NUM_EXPERTS = 2
                NET_CONFIG = {'0': 500, '1': 550, '2': 500}
                DROPOUT_PROB = 0.0
            elif CASE == 'bureau_sampled':
                NUM_EXPERTS = 2
                NET_CONFIG = {'0': 600, '1': 550, '2': 500}
                DROPOUT_PROB = 0.0
            print(f'======={c}=======')
            keys = get_all_unique_keys()

            # Statistics for point evaluation
            compared_idxes = get_random_point_idxes(len(keys))

            model_overheads = get_model_overheads(compared_idxes, keys, device, relative)
            noise_overheads = get_noise_overheads(compared_idxes, relative)
            missing_datas = get_single_noise_missing_data(compared_idxes)

            avg_model_overhead = sum(model_overheads) / len(model_overheads)
            avg_noise_overhead = sum(noise_overheads) / len(noise_overheads)
            avg_missing_datas = sum(missing_datas) / len(missing_datas)

            max_model_overhead = max(model_overheads)
            max_noise_overhead = max(noise_overheads)
            max_missing_datas = max(missing_datas)

            point_results[c] = (
            avg_model_overhead, avg_noise_overhead, max_model_overhead, max_noise_overhead, avg_missing_datas, max_missing_datas)

            # Statistics for interval evaluation
            compared_idx_pairs = get_random_interval_idxes(len(keys))

            model_overheads = get_model_overheads_for_intervals(compared_idx_pairs, keys, device, relative)
            noise_overheads = get_noise_overheads_for_intervals(compared_idx_pairs, relative)
            missing_datas = get_single_noise_missing_data_for_intervals(compared_idx_pairs)

            avg_model_overhead = sum(model_overheads) / len(model_overheads)
            avg_noise_overhead = sum(noise_overheads) / len(noise_overheads)
            avg_missing_datas = sum(missing_datas) / len(missing_datas)

            max_model_overhead = max(model_overheads)
            max_noise_overhead = max(noise_overheads)
            max_missing_datas = max(missing_datas)

            interval_results[c] = (
                avg_model_overhead, avg_noise_overhead, max_model_overhead, max_noise_overhead, avg_missing_datas,
                max_missing_datas)

        df = pd.DataFrame.from_dict(point_results, orient='index',
                                    columns=['Avg Model Overhead', 'Avg Noise Overhead', 'Max Model Overhead',
                                             'Max Noise Overhead', 'Avg Missing Data', 'Max Missing Data'])
        df.to_csv("overhead/" + ("relative" if relative else "absolute") + "_point_overhead.csv")

        df = pd.DataFrame.from_dict(interval_results, orient='index',
                                    columns=['Avg Model Overhead', 'Avg Noise Overhead', 'Max Model Overhead',
                                             'Max Noise Overhead', 'Avg Missing Data', 'Max Missing Data'])
        df.to_csv("overhead/" + ("relative" if relative else "absolute") + "_interval_overhead.csv")

if __name__ == '__main__':
    evaluate_overhead()