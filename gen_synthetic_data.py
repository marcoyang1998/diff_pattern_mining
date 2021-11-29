import pandas as pd
import numpy as np
from random import choice
import scipy.stats as stats

import string
from tqdm import tqdm
import json
import os
from config import logger

all_char = string.ascii_letters + string.digits


def rand_str():
    return "".join(choice(all_char) for _ in range(8))


def create_column_value_pool(col_num, min_cardinality, max_cardinality):

    logger.info("generate column value pool")

    pool = []

    for i in range(col_num):

        col_pool = set()
        cardinality = np.random.randint(min_cardinality, max_cardinality)

        for c in range(cardinality):
            value = rand_str()
            if value in col_pool:
                continue
            else:
                col_pool.add(value)
        pool.append(list(col_pool))

    return pool


def generate_root_cause(col_num, root_cause_length, cols_val_pool):

    logger.info("generate root cause")

    root_cause = {}

    root_cause_col_index = np.random.choice(col_num, size=root_cause_length, replace=False)

    for RC_col_index in root_cause_col_index:
        root_cause_val = np.random.choice(cols_val_pool[RC_col_index], size=1)[0]
        root_cause[RC_col_index] = root_cause_val

    logger.info(f"root cause:{root_cause}")

    return root_cause


def generate_attribute_data(col_num, row_num, cols_value_pool):

    logger.info("generate attribute data")

    data = pd.DataFrame()
    for col_index in tqdm(range(col_num)):
        data["col_"+str(col_index)] = np.random.choice(cols_value_pool[col_index], size=row_num, replace=True)

    return data


def generate_value_data(row_num, mu, var):

    logger.info("generate value data")

    value = np.floor(np.random.normal(mu, var, size=row_num))

    return value


def generate_failrate_data(row_num, mu, var):

    assert 0 < mu < 1, "out of prob range"

    lower = 0.0
    upper = 1.0

    failrate = stats.truncnorm.rvs((lower - mu) / var, (upper - mu) / var, loc=mu, scale=var, size=row_num)

    return failrate


def inject_root_cause_to_data(data, root_cause, root_cause_emerge_prob,
                             RC_failrate_mu, RC_failrate_var,
                             non_RC_failrate_mu, non_RC_failrate_var):

    logger.info("inject root cause to data")

    row_num = len(data)

    data["Failrate"] = generate_failrate_data(row_num, non_RC_failrate_mu, non_RC_failrate_var)

    root_cause_row_ids = np.random.choice(range(row_num), replace=False, size=int(root_cause_emerge_prob * row_num))
    root_cause_row_num = len(root_cause_row_ids)
    data["Failrate"].loc[root_cause_row_ids] = generate_failrate_data(root_cause_row_num, RC_failrate_mu, RC_failrate_var)

    data["Failed"] =data["Total"] * data["Failrate"]
    data["Failed"] = data["Failed"].apply(int)

    for rc_col, rc_val in root_cause.items():
        data["col_"+str(rc_col)].loc[root_cause_row_ids] = str(rc_val)

    return data


def transform_weighted_aggregated_data(data):

    logger.info("transform weighted aggregated data")

    data = data[data["Total"] > 0]

    data["Success"] = data['Total'] - data['Failed']
    data["label"] = data.apply(lambda x: [0, 1], axis=1)

    data = data.explode("label")

    data["weight"] = data.apply(lambda x: x["Failed"] if x["label"] == 1 else x["Success"], axis=1)

    data.drop(["Total", "Failed", "Success", "Failrate"], axis=1, inplace=True)
    data.reset_index(inplace=True, drop=True)

    return data


def generate_data(col_num, row_num, min_cardinality, max_cardinality, root_cause_emerge_prob,
                  value_mu, value_var,
                  RC_failrate_mu, RC_failrate_var,
                  non_RC_failrate_mu, non_RC_failrate_var, root_cause_length):

    cols_val_pool = create_column_value_pool(col_num, min_cardinality, max_cardinality)
    root_cause = generate_root_cause(col_num, root_cause_length, cols_val_pool)

    data = generate_attribute_data(col_num, row_num, cols_val_pool)

    data["Total"] = generate_value_data(row_num, value_mu, value_var)

    data = inject_root_cause_to_data(data, root_cause, root_cause_emerge_prob,RC_failrate_mu, RC_failrate_var, non_RC_failrate_mu, non_RC_failrate_var)

    data = transform_weighted_aggregated_data(data)

    root_cause_res = []
    for key, val in root_cause.items():
        root_cause_res.append(("col_"+str(key), str(val)))
    root_cause_res = tuple(root_cause_res)

    return data, root_cause_res


if __name__ == '__main__':
    #data_path = "data/col10_row1000_RCLen2_0.csv"
    #parse_synthetic_data(data_path)
    data_dir = 'data'
    label_dir = 'label'
    col_num_list = [10]
    row_num_list = [1e4]
    root_cause_length_list = [4]
    min_cardinality = 5
    max_cardinality = 20
    root_cause_emerge_prob = 0.2
    value_mu = 500
    value_var = 100
    RC_failrate_mu = 0.8
    RC_failrate_var = 0.2
    non_RC_failrate_mu = 0.1
    non_RC_failrate_var = 0.1

    dataset_num = 1

    for col_num in col_num_list:
        for row_num in row_num_list:
            for root_cause_length in root_cause_length_list:

                row_num = int(row_num)

                data_category = "col" + str(col_num) + "_row" + str(row_num) + "_RCLen" + str(root_cause_length)

                for i in range(dataset_num):

                    data_name = data_category + "_" + str(i)

                    logger.info(f"data name: {data_name}")

                    data, root_cause = generate_data(col_num, row_num, min_cardinality, max_cardinality, root_cause_emerge_prob,
                                  value_mu, value_var,
                                  RC_failrate_mu, RC_failrate_var,
                                  non_RC_failrate_mu, non_RC_failrate_var, root_cause_length)

                    data.to_csv(os.path.join(data_dir, data_name+".csv"), index=False)

                    with open(os.path.join(label_dir, data_name+"_root_cause.json"), "w") as f:
                        json.dump(root_cause, f)
    logger.info(f"Finish generating {dataset_num} dataset!")
