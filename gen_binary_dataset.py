import pandas as pd
import numpy as np
import random
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--num_pattern', type=int, default=2, help='How many patterns')
parser.add_argument('--num_cols', type=int, default=20, help='How many features')
parser.add_argument('--num_trans', type=int, default=10000, help='How many transactions')
parser.add_argument('--min_pattern_len', type=int, default=2, help='Minimum length of the pattern')
parser.add_argument('--max_pattern_len', type=int, default=5, help='Maximum length of the pattern')
parser.add_argument('--noise_ratio', type=float, default=0.05, help="Noise/signal ratio")
parser.add_argument('--output_folder', type=str, default='data', help="Where to store the generated dataset and pattern")


def generate_random_pattern(num_cols, pattern_length):
    return np.random.choice(num_cols, pattern_length, replace=False)


def generate_transaction_with_pattern(num_trans, num_cols, pattern):
    data = np.zeros((num_trans, num_cols))
    for i in range(num_trans):
        num_features = np.random.choice(range(2,num_cols-2), 1, replace=False)[0]
        select = np.random.choice(range(num_cols), num_features , replace=False)
        data[i, select] = 1
    data[:, pattern] = 1.0
    return data


def generate_transaction_without_pattern(num_trans, num_cols, pattern_list):
    data = np.zeros((num_trans, num_cols))

    for i in range(num_trans):
        num_features = np.random.choice(range(2,int(0.8*num_cols)), 1, replace=False)[0]
        select = np.random.choice(range(num_cols), num_features , replace=False)
        data[i, select] = 1.0
        for pattern in pattern_list:
            pattern_len = len(pattern)
            num_remove_pattern = np.random.choice(range(1,pattern_len+1), 1)[0]
            remove_pattern = np.random.choice(range(pattern_len), num_remove_pattern, replace=False)
            data[i, pattern[remove_pattern]] = 0.0

    return data


def add_noise(dataset, noise_ratio=0.01):
    num_trans, num_cols = dataset.shape
    dirty_ind = np.random.choice(range(num_trans), int(noise_ratio*num_trans), replace=False)
    dataset[dirty_ind] = ~dataset[dirty_ind]
    return dataset


def generate_binary_dataset(num_pattern, num_trans, num_cols, min_pattern_len, max_pattern_len, noise_ratio=0.01, ratio=0.5):
    #pos_num = int(num_trans*ratio)
    #neg_num = int(num_trans*(1-ratio))
    num_each_pattern = int(num_trans/(num_pattern+1))
    num_background = num_trans - num_each_pattern*num_pattern
    pattern_list = []
    pattern_data_list = []
    for i in range(num_pattern):
        pattern_length = np.random.choice(range(min_pattern_len, max_pattern_len+1), 1)
        pattern = generate_random_pattern(num_cols, pattern_length)
        pattern_list.append(pattern)
        pos_data = generate_transaction_with_pattern(num_each_pattern, num_cols, pattern)
        pattern_data_list.append(pos_data)

    pos_data = np.concatenate(pattern_data_list, axis=0)
    neg_data = generate_transaction_without_pattern(num_background, num_cols, pattern_list)
    dataset = np.concatenate((pos_data, neg_data), axis=0).astype(bool)
    dataset = add_noise(dataset, noise_ratio=noise_ratio)
    df = pd.DataFrame(dataset)
    df.columns = ["col_" + str(i) for i in range(num_cols)]
    labels = np.ones(num_trans).astype(bool)
    labels[-num_background:] = False
    df['label'] = labels

    for i, pattern in enumerate(pattern_list):
        has_pattern = neg_data[:, pattern].sum(axis=-1)
        rate = (has_pattern == len(pattern)).mean()
        print(f'{rate} of background data has pattern {i+1}')
    #has_pattern = (dataset[:,pattern] == True).sum(axis=-1)
    #pos_rate = (has_pattern[:pos_num] == len(pattern)).mean()
    #neg_rate = (has_pattern[pos_num:] == len(pattern)).mean()
    #print(f'Pos data: {pos_rate} contains the desired pattern')
    #print(f'Neg data: {neg_rate} contains the desired pattern')

    return df, pattern_list


def analyse_dataset(df):
    pass


if __name__ == '__main__':
    args = parser.parse_args()
    num_pattern = args.num_pattern
    num_trans = args.num_trans
    num_cols = args.num_cols
    min_pattern_len = args.min_pattern_len
    max_pattern_len = args.max_pattern_len
    noise_ratio = args.noise_ratio
    output_folder = args.output_folder

    file_name = f'{output_folder}/binary_{num_trans}trans_{num_cols}cols_{num_pattern}pattern_{noise_ratio}noise.csv'
    df, pattern_list = generate_binary_dataset(num_pattern=num_pattern,num_trans=num_trans, num_cols=num_cols,
                                          min_pattern_len=min_pattern_len, max_pattern_len=max_pattern_len,
                                          noise_ratio=noise_ratio)
    df.to_csv(file_name, index=False)
    print(f'Binary dataset stored at {file_name}')
    pattern_path = file_name.replace('.csv', '.txt')
    with open(pattern_path, 'w') as f:
        for p in pattern_list:
            pattern = [str(x) for x in p]
            f.write(' '.join(pattern) + '\n')
    print(f'Pattern stored at {pattern_path}')