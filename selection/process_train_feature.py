import copy
import json
import logging
import pickle
import sys
import os
import re
import math
from pathlib import Path
cur_path = Path(__file__).resolve().parent.parent
print(cur_path)
sys.path.append(cur_path.as_posix())
import time
import ast 
import random
import copy
from selection.benchmark import Benchmark
from selection.compare_algorithms.whole_compare import WholeCompare
from selection.dbms.hana_dbms import HanaDatabaseConnector
from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.query_generator import QueryGenerator
from selection.table_generator import TableGenerator
from selection.workload import Workload
from selection.utils import min_records_number
from selection.cost_evaluation import CostEvaluation
from selection.db_openbox import BlackBox
from BO.openbox_clapboard import ClapboardOpenbox
import torch

def read_config_dicts(log_file_path = '/data2/fray/index_selection_evaluation/train_log/train_tpch_100.log'):

    # Regular expression to match the arrays
    array_regex = re.compile(r'INFO:root:Gray code X: (\[\[.*?\])\s+INFO:root:Binary code:')

    # This list will hold all of the matched arrays
    matched_arrays = []

    # Open the log file and read line by line
    with open(log_file_path, 'r') as file:
        file_content = file.read()
        # Find all matches of the regular expression
        matches = array_regex.findall(file_content)
        for match in matches:
            # Append the matched array string, if you need it in list/tuple form, you'll have to parse it further
            try:
                match = ast.literal_eval(match)
            except ValueError as e:
                print(f"Error converting string to array: {e}")
            matched_arrays.append(match)
    
    return matched_arrays

config_dicts = read_config_dicts()[0:1280]
configs = []
# black_box = BlackBox(config_file = 'benchmark_results/fray_op/config_tpcds.json')
black_box = BlackBox(config_file = 'benchmark_results/fray_op/config_tpch100.json')
# black_box = BlackBox(config_file = 'benchmark_result s/fray_op/config_tpch.json')
# black_box = BlackBox(config_file = 'benchmark_results/fray_op/config_wikimedia.json', test = 1)
# black_box = BlackBox(config_file = 'benchmark_results/fray_op/config_test_join.json')
selection_algorithm = ClapboardOpenbox(black_box, random = True, gray = True, prf = True, max_runs = 2560, train=True, whitebox=True)
block_num = black_box.get_max_partition_number()
for config in config_dicts:
    print(config)
    config_dict = {}
    p_code = selection_algorithm.partition_tuples_to_code(config[0])
    for block_id in range(block_num):
        config_dict[f'x{block_id}'] = config[block_id+1]
    if selection_algorithm.dim_configs > 50:
        p_len = p_len = math.ceil((selection_algorithm.dim_configs-1) / 49)
        p_remain = (selection_algorithm.dim_configs-1) % 49
        binary_string = bin(p_code)[2:]  # 去掉开头的 '0b'
        # 在二进制字符串的开头不足 250 位时填充零
        padded_binary_string = binary_string.zfill(selection_algorithm.dim_configs)
        reversed_binary_string = padded_binary_string[::-1]
        res = 0
        power_of_2 = 1  
        # 从低位到高位依次取出连续的 50 位二进制数
        for i in range(0, selection_algorithm.dim_configs, 49):
            binary_chunk = reversed_binary_string[i:i+49]
            binary_chunk = binary_chunk[::-1]
            decimal_chunk = int(binary_chunk, 2)
            config_dict[f'p_{i // 49}'] = decimal_chunk
            res += decimal_chunk * power_of_2
            power_of_2 *= 2** 49
        assert res == p_code, "wrong"
    print(config_dict)
    configs.append(selection_algorithm.get_config_by_dic(config_dict))
for i in range(len(configs)):   
    query_features, data_features, index_features = selection_algorithm.white_box_model.get_features(configs[i])
    p = data_features[1][0]
    p = p.tolist()
    p = int(''.join(map(str, p)))
    p_code = selection_algorithm.partition_tuples_to_code(config_dicts[i][0])
    if p != p_code:
        print(f"{p} VS {p_code}")

print("finish process")

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       