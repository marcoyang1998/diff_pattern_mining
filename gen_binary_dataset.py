import pandas as pd
import numpy as np
import random


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


def generate_transaction_without_pattern(num_trans, num_cols, pattern):
    #data = np.random.randint(2, size=(num_trans, num_cols))
    data = np.zeros((num_trans, num_cols))
    pattern_len = len(pattern)
    for i in range(num_trans):
        num_features = np.random.choice(range(2,num_cols-2), 1, replace=False)[0]
        select = np.random.choice(range(num_cols), num_features , replace=False)
        data[i, select] = 1.0

        num_remove_pattern = np.random.choice(range(1,pattern_len), 1)[0]
        remove_pattern = np.random.choice(range(pattern_len), num_remove_pattern, replace=False)
        data[i, pattern[remove_pattern]] = 0.0

    return data


def add_noise(dataset, noise_ratio=0.01):
    num_trans, num_cols = dataset.shape
    dirty_ind = np.random.choice(range(num_trans), int(noise_ratio*num_trans), replace=False)
    dataset[dirty_ind] = ~dataset[dirty_ind]
    return dataset


def generate_binary_dataset(num_trans, num_cols, pattern_length, noise_ratio=0.01, ratio=0.5):
    pos_num = int(num_trans*ratio)
    neg_num = int(num_trans*(1-ratio))
    pattern = generate_random_pattern(num_cols, pattern_length)
    pos_data = generate_transaction_with_pattern(pos_num, num_cols, pattern)
    neg_data = generate_transaction_without_pattern(neg_num, num_cols, pattern)
    dataset = np.concatenate((pos_data, neg_data), axis=0).astype(bool)
    dataset = add_noise(dataset, noise_ratio=noise_ratio)
    df = pd.DataFrame(dataset)
    df.columns = ["col_" + str(i) for i in range(num_cols)]
    labels = np.zeros(pos_num+neg_num).astype(bool)
    labels[:pos_num] = True
    df['label'] = labels

    has_pattern = (dataset[:,pattern] == True).sum(axis=-1)
    pos_rate = (has_pattern[:pos_num] == len(pattern)).mean()
    neg_rate = (has_pattern[pos_num:] == len(pattern)).mean()
    print(f'Pos data: {pos_rate} contains the desired pattern')
    print(f'Neg data: {neg_rate} contains the desired pattern')

    return df, pattern


def analyse_dataset(df):
    pass


if __name__ == '__main__':
    num_trans = 1e5
    num_cols = 20
    pattern_length = 4
    noise_ratio = 0.05
    file_name = f'data/binary_{num_trans}trans_{num_cols}cols_{pattern_length}pl_{noise_ratio}noise.csv'
    df, pattern = generate_binary_dataset(num_trans=num_trans, num_cols=num_cols, pattern_length=pattern_length)
    df.to_csv(file_name, index=False)
    pattern = [str(x) for x in pattern]
    with open(file_name.replace('.csv', '.txt'), 'w') as f:
        f.write(' '.join(pattern))