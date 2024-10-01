import numpy as np
from math import ceil
import json

CRYPT = False

#Generate one-sided laplace noise
def gen_laplace_once(e, k):
    u = np.random.geometric(1 - pow(2.713, -e), k)
    v = np.random.geometric(1 - pow(2.713, -e), k)
    return abs(u - v + np.full(k, 7)) if not CRYPT else u - v

def gen_negative_noise(e, k):
    """
    Generates negative Laplace noise with a mean of -10.

    Parameters:
    - e (float): Parameter for Laplace noise generation.
    - k (int): Number of noise values to generate.

    Returns:
    - (numpy.ndarray): Array of negative noise values.
    """
    u = np.random.geometric(1 - pow(2.713, -e), k)
    v = np.random.geometric(1 - pow(2.713, -e), k)
    return -abs(u - v - np.full(k, 7))


binnum = 16

def generate_histogram(file_path, attribute, filter_type=None, filter_value=None, binnum=8):
    """
    Generates a histogram for a given attribute from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.
    - attribute (str): Attribute/column name for which histogram is needed.
    - filter_type (str, optional): Column name to filter the data (e.g. 'type'). Default is None.
    - filter_value (str, optional): Value to filter the data by (e.g. 'DISPONENT'). Default is None.
    - binnum (int, optional): Number of bins for the histogram. Default is 32.

    Returns:
    - hist (list): Histogram counts.
    """

    # Load the CSV data
    data = pd.read_csv(file_path)

    # Filter the data if filter_type and filter_value are specified
    if filter_type and filter_value:
        data = data[data[filter_type] == filter_value]

    binnum = np.unique(data[attribute]).shape[0]

    # Determine min, max and bin width (rounded up)
    min_val = int(data[attribute].min())
    max_val = int(data[attribute].max())
    bin_width = ceil((max_val - min_val) / binnum)

    # Compute bin edges
    bin_edges = [min_val + i*bin_width for i in range(binnum+1)]  # We need binnum+1 edges for binnum bins

    # Compute histogram
    hist, _ = np.histogram(data[attribute], bins=bin_edges)

    return hist, binnum


def noisy_histogram(file_path, attribute, e, filter_type=None, filter_value=None, binnum=8):
    """
    Generates a noisy histogram using Laplace noise for a given attribute from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.
    - attribute (str): Attribute/column name for which histogram is needed.
    - e (float): Parameter for Laplace noise generation.
    - filter_type (str, optional): Column name to filter the data (e.g. 'type'). Default is None.
    - filter_value (str, optional): Value to filter the data by (e.g. 'DISPONENT'). Default is None.
    - binnum (int, optional): Number of bins for the histogram. Default is 32.

    Returns:
    - noisy_histogram (list): Noisy histogram counts.
    """

    # Generate the histogram

    histogram, binnum = generate_histogram(file_path, attribute, filter_type, filter_value, binnum)

    # Convert the histogram to a numpy array
    histogram = np.array(histogram)

    # Add Laplace noise
    noisy_histogram = histogram + gen_laplace_once(e, binnum)

    return noisy_histogram.tolist()

def noisy_histogram_negative(file_path, attribute, e, filter_type=None, filter_value=None, binnum=8):
    """
    Generates a noisy histogram using negative Laplace noise for a given attribute from a CSV file.
    The histogram counts are ensured to be non-negative.

    Parameters:
    - file_path (str): Path to the CSV file.
    - attribute (str): Attribute/column name for which histogram is needed.
    - e (float): Parameter for Laplace noise generation.
    - filter_type (str, optional): Column name to filter the data (e.g. 'type'). Default is None.
    - filter_value (str, optional): Value to filter the data by (e.g. 'DISPONENT'). Default is None.
    - binnum (int, optional): Number of bins for the histogram. Default is 8.

    Returns:
    - noisy_histogram (list): Noisy histogram counts, with non-negative values.
    """

    # Generate the histogram
    histogram, binnum = generate_histogram(file_path, attribute, filter_type, filter_value, binnum)

    # Convert the histogram to a numpy array
    histogram = np.array(histogram)

    # Add negative Laplace noise
    noisy_histogram = histogram + gen_negative_noise(e, binnum)

    # Ensure non-negative histogram counts
    noisy_histogram = np.clip(noisy_histogram, 0, None)

    return noisy_histogram.tolist()


def compute_max_frequency(file_path, attribute, filter_type=None, filter_value=None):
    """
    Computes the exact maximum frequency of values under the specified attribute.

    Parameters:
    - file_path (str): Path to the CSV file.
    - attribute (str): Attribute/column name for which max frequency is computed.
    - filter_type (str, optional): Column name to filter the data (e.g. 'type'). Default is None.
    - filter_value (str, optional): Value to filter the data by (e.g. 'DISPONENT'). Default is None.

    Returns:
    - max_freq (int): Exact maximum frequency.
    """

    # Load the CSV data
    data = pd.read_csv(file_path)

    # Filter the data if filter_type and filter_value are specified
    if filter_type and filter_value:
        data = data[data[filter_type] == filter_value]

    # Compute max frequency
    max_freq = data[attribute].value_counts().max()

    return max_freq

def noisy_max_frequency(file_path, attribute, e, filter_type=None, filter_value=None):
    """
    Computes the noisy maximum frequency of values under the specified attribute.

    Parameters:
    - file_path (str): Path to the CSV file.
    - attribute (str): Attribute/column name for which max frequency is computed.
    - e (float): Parameter for Laplace noise generation.
    - filter_type (str, optional): Column name to filter the data (e.g. 'type'). Default is None.
    - filter_value (str, optional): Value to filter the data by (e.g. 'DISPONENT'). Default is None.

    Returns:
    - noisy_max_freq (int): Noisy maximum frequency.
    """

    # Compute the max frequency
    max_freq = compute_max_frequency(file_path, attribute, filter_type, filter_value)

    # Noisy max (TODO - this is bugged, should add noise then find max)
    noisy_max_freq = max_freq + np.random.exponential(scale=1/e, size=1)[0]
    return int(noisy_max_freq)

def compute_cdf_indexes(positive_noisy_histogram, negative_noisy_histogram):
    """
    Computes CDF indexes using the true histogram and both positive and negative noisy histograms.

    Parameters:
    - positive_noisy_histogram (list of int): Noisy histogram counts using positive noise.
    - negative_noisy_histogram (list of int): Noisy histogram counts using negative noise.

    Returns:
    - indexes (list of tuples): Ordered list of (lo_i, hi_i) index pairs, where lo_i is
                                derived from the negative noisy histogram and hi_i from
                                the positive noisy histogram, with hi_i capped by the sum of the true histogram.
    """

    # Calculate the CDF for the positive noisy histogram
    cdf_positive = np.cumsum(positive_noisy_histogram)

    # Calculate the CDF for the negative noisy histogram
    cdf_negative = np.cumsum(negative_noisy_histogram)

    # Total count in the true histogram
    C = sum(list(positive_noisy_histogram))

    # Initialize the list to store index pairs
    indexes = []

    # Calculate lo_i and hi_i for each bin, capping hi_i with C
    for i in range(len(positive_noisy_histogram)):
        if (i == 0):
            lo_i = 0
            hi_i = min(cdf_positive[i], C)
        else:
            lo_i = cdf_negative[i-1]
            hi_i = min(cdf_positive[i], C)
        indexes.append((lo_i, hi_i))

    return indexes

def generate_synopsis(csv_path, target_attribute, epsilon=1, filter_attribute=None, filter_value=None):
    noisy_hist = noisy_histogram(csv_path, target_attribute, epsilon, filter_type=filter_attribute, filter_value=filter_value)
    noisy_neg_hist = noisy_hist if CRYPT else noisy_histogram_negative(csv_path, target_attribute, epsilon, filter_type=filter_attribute, filter_value=filter_value)
    indexes = compute_cdf_indexes(noisy_hist, noisy_neg_hist)
    noisy_mf = noisy_max_frequency(csv_path, target_attribute, epsilon, filter_type=filter_attribute, filter_value=filter_value)

    return {
        "noisy_histogram": noisy_hist,
        "noisy_negative_histogram": noisy_neg_hist,
        "indexes": indexes,
        "noisy_max_frequency": noisy_mf
    }

import pandas as pd

def exp_synop_generation(epsilon=1):
    cases = ['uniform3', 'uniform4', 'uniform5', 'uniform6',
             'lognormal3', 'lognormal4', 'lognormal5','lognormal6',
             'trans_sampled',
             'bureau_sampled_250k', 'bureau_sampled']
    for c in cases:
        if not c.startswith('trans_sampled') and not c.startswith('bureau_sampled'):
            attribute = 'key'
        elif c.startswith('bureau_sampled'):
            attribute = 'SK_ID_BUREAU'
        elif c == 'trans_sampled':
            attribute = 'account_id'
        else:
            raise ValueError(f"Wrong case.")
        synopsis = generate_synopsis(f'dataset/{c}.csv', attribute, epsilon=epsilon)
        with open(('crypt/' if CRYPT else 'noise/') + f'{c}.json', 'w') as file:
            json.dump(synopsis, file, default=default_dump)


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

if __name__ == '__main__':
    exp_synop_generation(epsilon=1)
