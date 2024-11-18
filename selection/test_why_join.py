import copy
import json
import logging
import pickle
import sys
import os
from pathlib import Path
cur_path = Path(__file__).resolve().parent.parent
print(cur_path)
sys.path.append(cur_path.as_posix())
import time 
import random
import copy
from selection.db_openbox import BlackBox

def test_why_join():
    random.seed(1)
    black_box = BlackBox(1, config_file= "benchmark_results/fray_op/config_test_join.json")
    partition_max  = black_box.get_max_partition_number()
    attributes = black_box.get_attributes_number()
    partition_number = 6
    index_number = 6
    partition_codes = [random.randint(0, 2**(partition_max -1)-1) for i in range(partition_number)]
    indexes_space = [2 ** index_number for index_number in black_box.get_attributes_number()]
    indexes_code = [[random.randint(0, index_space-1) for index_space in indexes_space] for i in range(index_number)]
    res = []
    for i in range(partition_number):
        partition_code = partition_codes[i]
        for j in range(index_number):
            config_list = [black_box.code_to_partition_tuples(partition_code, partition_max)]+indexes_code[j]
            index_combination_size,cost = black_box.run_tupels(config_list, partition_code)
            res.append(cost)
    for i in range(len(res)):
        if i % index_number == 0:
            print("----------------")
        print(res[i])

def test_index_disparity():
    random.seed(1)
    black_box = BlackBox(1, config_file= "benchmark_results/fray_op/config_index_disparity.json")
    partition_max  = black_box.get_max_partition_number()
    attributes = black_box.get_attributes_number()
    partition_number = 6
    index_number = 6
    partition_codes = [random.randint(0, 2**(partition_max -1)-1) for i in range(partition_number)]
    indexes_space = [2 ** index_number for index_number in black_box.get_attributes_number()]
    indexes_code = [[random.randint(0, index_space-1) for index_space in indexes_space] for i in range(index_number)]
    partition_code = 0
    res = black_box.test_index_disparity(partition_code)
    for block_id,index_cost in res.items():
            print(block_id)
            for index,cost in index_cost.items():
                print(index, cost)


test_why_join()