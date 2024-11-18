import numpy as np
from openbox import space as sp, Optimizer
import os
import logging
logging.getLogger().setLevel(logging.INFO)
import sys
sys.path.append("/mydata/BO/index_selection_evaluation")
print(sys.path)
from index_selection_evaluation.selection.db_openbox import BlackBox

db_config = {
    'num_variables': 20,    #分区种类数
    'num_indices': 100, #索引种类数
    'budget': 200
}

def max_performance_within_partition(config: sp.Configuration):
    X = np.array([config[f'x{i}'] for i in range(db_config['num_variables'])])
    index_combination_size, cost = black_box.run()
    