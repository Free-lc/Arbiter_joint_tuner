import numpy as np
from openbox import space as sp, Optimizer
from ConfigSpace import Configuration
import math
import os
import logging
import copy
logging.getLogger().setLevel(logging.INFO)
import sys
sys.path.append("/mydata/fray/index_selection_evaluation")
print(sys.path)
from selection.db_openbox import BlackBox
from BO.whitebox import WhiteBoxModel as white_box_model
db_config = {
    'num_variables': 20,    #分区种类数
    'num_indices': 100, #索引种类数
    'budget': 600
}

import re

class ClapboardOpenbox():
    def __init__(
        self,
        # budget: float = 600,
        black_box: BlackBox = None,
        gray = True,
        prf = True,
        random = False,
        max_runs = 100,
        important_parameters = None,
        best_random_configs = None,
        prefilter = False,
        train = False,
        whitebox = False,
    ) -> None:
        self.black_box = black_box
        if self.black_box is None:
            self.black_box = BlackBox()
        self.prefilter = prefilter
        if self.prefilter:
            # Prefilter
            self.black_box.prefilter_workload()
        self.dim_configs = self.black_box.get_max_partition_number()
        self.indexes_space = [2 ** index_number for index_number in self.black_box.get_attributes_number()]  # 默认不会溢出
        self.budget = self.black_box.budget
        self.gray = gray
        self.prf = prf
        self.random = random
        self.train = train
        if "tpch" in self.black_box.database_name:
            cache_file = f'BO/cache_tpch_{self.black_box.config["scale_factor"]}.pkl'
        elif "tpcds" in self.black_box.database_name:
            cache_file = f'BO/cache_tpcds_{self.black_box.config["scale_factor"]}.pkl'
        elif "wikimedia" in self.black_box.database_name:
            cache_file = f'BO/cache_wikimedia_{self.black_box.config["scale_factor"]}.pkl'
        elif "test" in self.black_box.database_name:
            cache_file = f'BO/cache_tpch_{self.black_box.config["scale_factor"]}.pkl'
        else:
            assert 0, "undefined benchmarks!!!"

        self.partition_code_range = partition_code_range = 2 ** (self.dim_configs - 1)
        # max range of 'p' config is 2 ^ 49, if 'p' range over maximum range, split 'p' range
        if self.dim_configs > 50:
            self.p_over = True
            self.p_len = p_len = math.ceil((self.dim_configs-1) / 49)
            p_remain = (self.dim_configs-1) % 49
            self.params = {}
            for i in range(p_len):
                low = 0
                upper = 2 ** 49 - 1
                if i == p_len -1:
                   upper = 2 ** p_remain - 1  
                self.params[f'p_{i}'] = (low, upper, upper)
        else:
            self.p_over = False
            self.params = {'p': (0, partition_code_range - 1, partition_code_range - 1)}    # (low, upper, dafault)
        for i in range(self.dim_configs):
            self.params[f'x{i}'] = (0, self.indexes_space[i] - 1, 0)
        self.space = sp.Space()
        # incremental config selection
        if important_parameters is not None and best_random_configs is not None:
            new_params = copy.deepcopy(self.params)
            for name, para in new_params.items():
                if name not in important_parameters and 'p' not in name:
                    # fix unimportant parameters
                    self.params[name] = [0]    # fixed value
                    # self.params[name] = (best_random_configs[name], best_random_configs[name], best_random_configs[name])
        space_config = []
        for name, para in self.params.items():
            if important_parameters is not None and best_random_configs is not None and name not in important_parameters:
                space_config.append(sp.Categorical(name, para))
            else:
                if para[0] == para[1]:
                    # when it occurs, it means we have only one chose
                    space_config.append(sp.Categorical(name, [para[0]]))
                else:
                    space_config.append(sp.Int(name, *para))
                    # space_config.append(sp.Categorical(name, choices = list(range(para[0], para[1]+1)), default_value = para[2]))
        self.space.add_variables(space_config)
        # self.space.set_sample_condition(self.sample_condition)
        if whitebox:
            self.white_box_model = white_box_model(black_box = self.black_box, cache_file = cache_file, train=train)
        else:
            self.white_box_model = None
        self.opt = Optimizer(
            objective_function = self.obj_func,
            config_space = self.space,
            num_constraints = 1,
            num_objectives = 1,
            surrogate_type = 'prf' if self.prf else 'gp',
            acq_optimizer_type = 'auto',
            max_runs = max_runs,
            task_id = 'basic_search',
            logging_dir = 'logs_gray_clapboard_search',
            visualization = 'basic',
            initial_runs = 30 if not self.random else 0,
            random_state = 999,
            advisor_type = 'random' if self.random else 'default',
            white_box_model = self.white_box_model,
            train = train
        )

        self.history = None

    def get_config_by_dic(self, config_dict: dict):
        config = Configuration(self.space, config_dict)
        return config

    def sample_condition(self, config):
        if self.p_over:
            gray = 0
            for i in range(self.p_len):
                gray += ((2**49)**i) * config[f'p_{i}']
        else:           
            gray = config['p']
        if self.gray:
            binary = self.gray_to_binary(gray)
        else:
            binary = gray
        partition_tuples = self.code_to_partition_tuples(binary)
        X = []
        if self.gray:
            X = X + [self.gray_to_binary(config[f'x{i}']) for i in range(self.dim_configs)]
        else:
            X = X + [config[f'x{i}'] for i in range(self.dim_configs)]
        for partition_tuple in partition_tuples:
            start = partition_tuple[0]
            end = partition_tuple[1]
            if self.check_same_elements(X, start, end):
                continue
            else:
                return False
        return True


    def check_same_elements(self,lst, a, b):
        sub_list = lst[a:b]  # 切片获取子列表，注意索引从 0 开始，所以需要减 1
        return len(set(sub_list)) == 1

    def get_X_from_config(self, config: sp.Configuration):
        if self.p_over:
            gray = 0
            for i in range(self.p_len):
                gray += ((2**49)**i) * config[f'p_{i}']
        else:           
            gray = config['p']
        X_before = [self.code_to_partition_tuples(gray)]
        X_before = X_before + [config[f'x{i}'] for i in range(self.dim_configs)]
        logging.info(f'Gray code X: {X_before}')
        if self.gray:
            binary = self.gray_to_binary(gray)
        else:
            binary = gray
        partition_tuples = self.code_to_partition_tuples(binary)
        X = [partition_tuples]
        if self.gray:
            X = X + [self.gray_to_binary(config[f'x{i}']) for i in range(self.dim_configs)]
        else:
            X = X + [config[f'x{i}'] for i in range(self.dim_configs)]
        logging.info(f'Binary code: {X}')
        return X, binary
    
    def obj_func(self, config: sp.Configuration):

        X, binary = self.get_X_from_config(config)
        index_combination_size,cost = self.black_box.run_tupels(X, binary)
        logging.info(f'cost = {cost}, index_combination_size = {index_combination_size}')
        result = {
            'objectives': [cost, ],
            'constraints': [index_combination_size - self.budget, ]
        } 
        return result

    def run(self):
        self.history = self.opt.run()
        print(self.history)

    def get_important_parameters(self, rate = 1):
        #[0,1,0.2,0.7,....,]
        column_id = self.history.get_importance().field_names.index('Obj1 Importance')
        importances = [row[column_id] for row in self.history.get_importance().rows]
        parameter_id = self.history.get_importance().field_names.index('Parameter')
        parameters = [row[parameter_id] for row in self.history.get_importance().rows]
        important_idx = self._find_top_n_indices(importances, math.ceil(len(importances)*rate))
        important_parameters = [parameters[idx] for idx in important_idx]
        return important_parameters

    def _find_top_n_indices(self, lst, n):
        indices = {}
        for i, num in enumerate(lst):
            indices[i] = num

        top_n_indices = sorted(indices, key=indices.get, reverse=True)[:n]
        return top_n_indices

    def partition_tuples_to_code(self, partition_tuple_list: list, dim_configs: int = 250) -> int:
        # Initialize bit code
        code = 0
        assert dim_configs == self.dim_configs, "wrong data block number"
        # Sort the partition tuple list by the first element of each tuple
        partition_tuple_list = sorted(partition_tuple_list, key=lambda x: x[0])
        # Determine split points from partition tuples
        split_points = [end for (start, end) in partition_tuple_list[:-1]]
        # Calculate the code from the split points
        for sp in split_points:
            bit_position = self.dim_configs - sp - 1
            code |= 1 << bit_position
        return code

    def code_to_partition_tuples(self, code: int, dim_configs: int = 50):
        split_points = []
        partition_tuple_list = [(0,self.dim_configs)]
        pos = 0
        while code > 0:
            digit = code & 1
            if digit == 1:
                split_points.append(self.dim_configs - 1 - pos)
            code = code >> 1
            pos += 1
        for pt in split_points:
            new_list = []
            for partition_tuple in partition_tuple_list:
                start, end = partition_tuple
                if start < pt <= end:
                    new_list.append((start, pt))
                    new_list.append((pt, end))
                else:
                    new_list.append(partition_tuple)
            partition_tuple_list = new_list
                    
        return partition_tuple_list

    def gray_to_binary(self, gray):
        mask = gray >> 1
        while mask != 0:
            gray = gray ^ mask
            mask = mask >> 1
        return gray

    def get_optimal_configs(self):
        return self.history.get_incumbent_configs()

    def get_optimal_value(self):
        return self.history.get_incumbent_value()

    def get_optimal_history(self):
        return self.history.get_incumbents()

def extract_configs_from_log(log_file):
    values = []
    pattern = r'\|\s*(\w+)\s*\|\s*(\d+)\s*\|'   # 正则表达式模式，用于匹配参数和数值

    with open(log_file, 'r') as file:
        contents = file.read()

        matches = re.findall(pattern, contents)   # 在日志内容中查找匹配的模式
        for match in matches:
            param, value = match
            if param.startswith('x') or param == 'p':  # 只保留以'x'开头和等于'p'的参数值
                values.append(int(value))


    return values

def gray_to_binary(gray):
        mask = gray >> 1
        while mask != 0:
            gray = gray ^ mask
            mask = mask >> 1
        return gray

def test_single_config():
    optimal_configs = {
        'prf':[
            ([[(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 14), (14, 16), (16, 17), (17, 18), (18, 20)], 177683, 124647, 3673, 491226, 65626, 154065, 215990, 99123, 104610, 318745, 447840, 387555, 130197, 268364, 170346, 300302, 85235, 164122, 73224, 377805], 782382),
            ([[(0, 2), (2, 5), (5, 6), (6, 7), (7, 9), (9, 10), (10, 12), (12, 13), (13, 15), (15, 16), (16, 17), (17, 19), (19, 20)], 144253, 458156, 103859, 449583, 233949, 266561, 315666, 165816, 122664, 42006, 323728, 39580, 166502, 66912, 117365, 124410, 63315, 344096, 105045, 452931], 685789),
            ([[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 7), (7, 10), (10, 13), (13, 16), (16, 17), (17, 18), (18, 19), (19, 20)], 287240, 304907, 87674, 362865, 111353, 461376, 107453, 28864, 449749, 38916, 311065, 388021, 80258, 204721, 205817, 312330, 515155, 34140, 90452, 80224], 1036879),
            ([[(0, 2), (2, 3), (3, 6), (6, 7), (7, 9), (9, 10), (10, 11), (11, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20)], 454968, 338656, 507384, 501539, 42561, 102659, 211256, 40306, 34805, 70512, 456079, 11907, 57902, 11911, 54125, 74915, 289546, 119089, 175541, 450378], 735039)
        ],
        'prf_gray':[
            ([[(0, 2), (2, 3), (3, 4), (4, 6), (6, 7), (7, 10), (10, 14), (14, 16), (16, 18), (18, 20)], 128531, 520600, 204696, 334774, 511602, 400212, 418110, 233786, 32713, 380610, 51473, 87659, 484430, 309968, 263003, 511928, 262481, 104623, 394431, 349346], 158527),
            ([[(0, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 10), (10, 12), (12, 13), (13, 14), (14, 16), (16, 18), (18, 19), (19, 20)], 54035, 242932, 46125, 32369, 0, 66583, 244558, 129002, 126517, 259466, 18225, 111062, 9869, 0, 3985, 55985, 1939, 49625, 259890, 48318], 34718),
            ([[(0, 1), (1, 2), (2, 4), (4, 8), (8, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 18), (18, 20)], 511474, 301985, 228274, 346990, 183314, 519980, 322314, 496538, 237467, 459960, 448771, 223402, 155771, 3365, 376411, 80944, 492044, 19076, 120304, 471789], 642571),
            ([[(0, 3), (3, 5), (5, 7), (7, 8), (8, 10), (10, 13), (13, 15), (15, 16), (16, 18), (18, 20)], 198353, 230978, 167449, 35123, 281749, 3749, 316790, 490051, 14336, 302418, 359552, 335235, 336426, 128759, 40617, 342065, 304313, 212350, 479878, 276151], 128887)
        ],
        'gp':[
            ([[(0, 2), (2, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 13), (13, 16), (16, 17), (17, 19), (19, 20)], 51209, 121302, 475849, 161907, 275153, 438289, 248110, 139552, 441, 304605, 36784, 138226, 250025, 355659, 435068, 69863, 444092, 237083, 227061, 118220], 720717),
            ([[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 14), (14, 15), (15, 16), (16, 18), (18, 19), (19, 20)], 182848, 0, 468094, 199670, 432679, 369347, 9991, 381147, 151743, 142265, 61265, 341413, 279003, 159527, 385688, 203354, 35114, 79317, 391310, 237099], 523835),
            ([[(0, 1), (1, 7), (7, 8), (8, 9), (9, 13), (13, 16), (16, 18), (18, 19), (19, 20)], 81863, 387896, 89315, 15285, 478823, 61865, 403586, 134227, 57610, 486418, 476898, 475468, 284971, 441686, 432269, 186033, 345331, 89112, 359189, 52617], 269387),
            ([[(0, 1), (1, 3), (3, 4), (4, 10), (10, 15), (15, 16), (16, 18), (18, 20)], 261554, 272144, 369235, 249611, 264787, 443899, 520205, 126111, 71809, 100147, 426064, 229020, 503583, 58572, 167456, 65265, 212914, 490085, 380891, 139134], 360986)
        ],
        'gp_gray':[
            ([[(0, 1), (1, 3), (3, 4), (4, 6), (6, 8), (8, 9), (9, 10), (10, 11), (11, 14), (14, 15), (15, 16), (16, 17), (17, 19), (19, 20)], 255166, 148931, 404697, 267352, 419619, 414281, 273986, 83495, 147077, 33808, 202128, 42354, 413388, 454666, 460494, 179539, 152666, 38154, 131033, 411489], 489635),
            ([[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 7), (7, 9), (9, 11), (11, 12), (12, 13), (13, 16), (16, 18), (18, 19), (19, 20)], 224947, 493967, 115155, 366666, 293722, 329842, 479102, 316353, 126258, 115891, 356405, 453384, 0, 488275, 346427, 144791, 221242, 466652, 387379, 188334], 540462),
            ([[(0, 2), (2, 3), (3, 4), (4, 7), (7, 9), (9, 10), (10, 11), (11, 12), (12, 14), (14, 15), (15, 20)], 93113, 338228, 165394, 256899, 149521, 496756, 251306, 195628, 302240, 157906, 490174, 82568, 460201, 126856, 101538, 248688, 369238, 347060, 281486, 232545], 154728),
            ([[(0, 2), (2, 8), (8, 11), (11, 17), (17, 18), (18, 20)], 0, 349525, 437683, 349525, 192850, 64270, 349564, 418570, 349525, 0, 371248, 95667, 349525, 349525, 349525, 496220, 0, 349525, 0, 349525], 986501)
        ]
    }
    black_box = BlackBox()
    performances = []
    for experiment, configs in optimal_configs.items():
        total_performance = 0
        logging.info(f'evaluating [ {experiment} ]: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        for index, config in enumerate(configs):
            x, partition_code = config
            if(experiment == 'prf_gray' or experiment == 'gp_gray'):
                partition_code = gray_to_binary(partition_code)
            logging.info(f'{experiment}_{index}:')
            cost, index_size = black_box.run_tupels(x, partition_code)
            logging.info(f'cost = {cost}, index_combination_size = {index_size}')
            total_performance += cost
        performances.append(total_performance / 4)
        logging.info(f'avg_performance: {total_performance / 4}')

def ablation_experiment():
    value_map = {"prf_without_grey": 0, "prf_grey":0, "gp_without_grey":0, "gp_grey":0, "random":0}
    black_box = BlackBox()
    for i in range(3):
        co_random = ClapboardOpenbox(black_box = black_box, random = True)
        co_random.run()
        value_map["random"] += co_random.get_optimal_value()

        co_without_grey = ClapboardOpenbox(black_box = black_box, gray = False)
        co_without_grey.run()
        value_map["prf_without_grey"] += co_without_grey.get_optimal_value()
        co = ClapboardOpenbox(black_box = black_box)
        co.run()
        value_map["prf_grey"] += co.get_optimal_value()
        co_gp_without_grey = ClapboardOpenbox(black_box = black_box, gray = False, prf = False)
        co_gp_without_grey.run()
        value_map["gp_without_grey"] += co_gp_without_grey.get_optimal_value()
        co_gp = ClapboardOpenbox(black_box = black_box, gray = True, prf = False)
        co_gp.run()
        value_map["gp_grey"] += co_gp.get_optimal_value()        
    for key,value in value_map.items():
        print(f"{key} : {value}")

if __name__ == '__main__':
    #main()
    ablation_experiment()


    
