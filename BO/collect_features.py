import logging
import pickle
import sys
import math
import os
from pathlib import Path
from BO.whitebox import WhiteBoxModel as white_box_model
from selection.db_openbox import BlackBox
from BO.openbox_clapboard import ClapboardOpenbox

# black_box = BlackBox(config_file = 'benchmark_results/fray_op/config_tpcds.json')
black_box = BlackBox(config_file = 'benchmark_results/fray_op/config_tpch100.json')
# black_box = BlackBox(config_file = 'benchmark_results/fray_op/config_tpch.json')
# black_box = BlackBox(config_file = 'benchmark_results/fray_op/config_wikimedia.json', test = 1)
# black_box = BlackBox(config_file = 'benchmark_results/fray_op/config_test_join.json')
selection_algorithm = ClapboardOpenbox(black_box, random = True, gray = True, prf = True, max_runs = 2560, train=True, whitebox=True)
configs = []
block_num = black_box.get_max_partition_number()
for i in range(2 ** (block_num-1)):
    selection_algorithm.white_box_model.get_data_features(i)
    print(f"patition_code {i} has been cached")