import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def read_records():
    ret = []
    df = pd.read_csv('overhead/relative_point_overhead.csv')
    ret.append(df.to_dict(orient='index'))
    df = pd.read_csv('overhead/relative_interval_overhead.csv')
    ret.append(df.to_dict(orient='index'))
    df = pd.read_csv('overhead/absolute_point_overhead.csv')
    ret.append(df.to_dict(orient='index'))
    df = pd.read_csv('overhead/absolute_interval_overhead.csv')
    ret.append(df.to_dict(orient='index'))
    return ret

def view(results, point=True, show_multipliers=True, relative=False):
    c_values = list(results.keys())
    avg_model_overhead = [results[c]['Avg Model Overhead'] for c in c_values]
    avg_noise_overhead = [results[c]['Avg Noise Overhead'] for c in c_values]
    max_model_overhead = [results[c]['Max Model Overhead'] for c in c_values]
    max_noise_overhead = [results[c]['Max Noise Overhead'] for c in c_values]
    avg_missing_datas = [results[c]['Avg Missing Data'] for c in c_values]
    max_missing_datas = [results[c]['Max Missing Data'] for c in c_values]
    c_values = ['uniform3', 'uniform4', 'uniform5', 'uniform6',
             'lognormal3', 'lognormal4', 'lognormal5', 'lognormal6',
             'trans_sampled',
             'bureau_sampled_250k', 'bureau_sampled']

    print("Average missing data:", avg_missing_datas)
    print("Max missing data:", max_missing_datas)

    bar_width = 0.35
    index = np.arange(len(c_values))

    # 1st avg
    plt.figure(figsize=(12, 6))
    plt.bar(index, avg_model_overhead, bar_width, label='Learned Index Average Overhead')
    plt.bar(index + bar_width, avg_noise_overhead, bar_width, label='Random Noise Average Overhead')

    plt.yscale('log')

    plt.xlabel('Experiment')
    plt.ylabel('Average Overhead (log scale)')
    plt.title(('Relative ' if relative else 'Absolute ') + ('Point ' if point else 'Interval ') + 'Average Overhead: Learned Index vs Random Noise')
    plt.xticks(index + bar_width / 2, c_values, rotation=45, ha='right')
    plt.legend()

    if show_multipliers:
        for i in range(len(c_values)):
            values = [avg_model_overhead[i], avg_noise_overhead[i]]

            min_value = min(values)
            sorted_values = sorted(values)
            max_value = sorted_values[1]

            if max_value > min_value:
                factor = max_value / min_value
                max_index = values.index(max_value)
                plt.text(i + max_index * bar_width, max_value, f'{factor:.1f}x', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # 2nd image: max
    plt.figure(figsize=(12, 6))
    plt.bar(index, max_model_overhead, bar_width, label='Learned Index Max Overhead')
    plt.bar(index + bar_width, max_noise_overhead, bar_width, label='Random Noise Max Overhead')

    plt.yscale('log')

    plt.xlabel('Experiment')
    plt.ylabel('Max Overhead (log scale)')
    plt.title(('Relative ' if relative else 'Absolute ') + ('Point ' if point else 'Interval ') + 'Max Overhead: Learned Index vs Random Noise')
    plt.xticks(index + bar_width / 2, c_values, rotation=45, ha='right')
    plt.legend()

    if show_multipliers:
        for i in range(len(c_values)):
            values = [max_model_overhead[i], max_noise_overhead[i]]

            min_value = min(values)
            sorted_values = sorted(values)
            max_value = sorted_values[1]

            if max_value > min_value:
                factor = max_value / min_value
                max_index = values.index(max_value)
                plt.text(i + max_index * bar_width, max_value, f'{factor:.1f}x', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def draw(records):
    for i, result in enumerate(records):
        relative = i <= 1
        point = i % 2 == 0
        view(result, relative=relative, point=point, show_multipliers=True)


if __name__ == '__main__':
    draw(read_records())