import numpy as np
from openbox import space as sp, Optimizer
import os
import logging
logging.getLogger().setLevel(logging.INFO)
import sys
from pathlib import Path
cur_path = Path(__file__).resolve().parent.parent
print(cur_path)
sys.path.append(cur_path.as_posix())
from index_selection_evaluation.selection.db_openbox import BlackBox
db_config = {
    'num_variables': 20,    #分区种类数
    'num_indices': 100, #索引种类数
    'budget': 600
}

black_box = BlackBox()

def max_performance(config: sp.Configuration):
    X = np.array([config['p']])
    X = np.append(X, [gray_to_binary(config[f'x{i}']) for i in range(db_config['num_variables'])])
    index_combination_size,cost = black_box.run(X)
    logging.info(f'variables：{X}')
    logging.info(f'cost = {cost}, index_combination_size = {index_combination_size}')
    result = {
        'objectives': [cost, ],
        'constraints': [index_combination_size - db_config['budget'], ]
    } 

    return result

def gray_to_binary(self, gray):
        mask = gray >> 1
        while mask != 0:
            gray = gray ^ mask
            mask = mask >> 1
        return gray

def main():
    db_config['num_variables'] = black_box.get_max_partition_number()
    db_config['num_indices'] = 2 ** black_box.get_attribute_number()
    
    params = {'p': (1, db_config['num_variables'], 1)}

    for i in range(db_config['num_variables']):
        params[f'x{i}'] = (0, db_config['num_indices'] - 1, 0) 

    space = sp.Space()
    space.add_variables([
        sp.Int(name, *para) for name, para in params.items()
    ])

    opt = Optimizer(
        objective_function = max_performance,
        config_space = space,
        num_constraints = 1,
        num_objectives = 1,
        surrogate_type = 'prf',
        acq_optimizer_type = 'random_scipy',
        max_runs = 300,
        task_id = 'basic_search',
        logging_dir = 'logs_basic_search',
        visualization = 'basic'
    )


    history = opt.run()
    print(history)
    # history.visualize_html(
    #     open_html=True,
    #     show_importance=False,
    #     verify_surrogate=False,
    #     optimizer=opt
    # )

if __name__ == '__main__':
    main()
    #print(black_box.get_max_partition_number())