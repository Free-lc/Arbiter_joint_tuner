from selection.compare_algorithms.recommend_algorithm import SelectionAlgorithm
from BO.openbox_clapboard import ClapboardOpenbox
import math
import logging
class Fray(SelectionAlgorithm):
    def __init__(self, black_box, gray = True, prf = True, random = False, max_runs = 100, train=False, whitebox = False):
        SelectionAlgorithm.__init__(self,black_box)
        self.gray = gray
        self.prf = prf
        self.random = random
        self.max_runs = max_runs
        self.clapboardopenbox = ClapboardOpenbox(black_box = self.black_box, gray = self.gray, prf = self.prf, random = self.random, prefilter = True, max_runs=max_runs, train=train, whitebox=whitebox)

    def get_optimal_value(self):
        self.num_pages = self.black_box.get_max_partition_number()
        self.clapboardopenbox.run()
        clap_config = self.clapboardopenbox.get_optimal_configs()[0]
        X, binary = self.clapboardopenbox.get_X_from_config(clap_config)
        index_combination_size,cost = self.black_box.run_tupels(X, binary)
        return index_combination_size,cost

class FrayPreFilter(SelectionAlgorithm):
    def __init__(self, black_box, gray = True, prf = True, random = False, max_runs = 100, train = False, whitebox = False):
        SelectionAlgorithm.__init__(self,black_box)
        self.gray = gray
        self.prf = prf
        self.random = random
        self.max_runs = max_runs
        #set parameter prefilter to True
        self.clapboardopenbox = ClapboardOpenbox(black_box = self.black_box, gray = self.gray, prf = self.prf, random = self.random, prefilter = True, max_runs=self.max_runs, train=train, whitebox=whitebox)

    def get_optimal_value(self):
        self.num_pages = self.black_box.get_max_partition_number()
        self.clapboardopenbox.run()
        clap_config = self.clapboardopenbox.get_optimal_configs()[0]
        X, binary = self.clapboardopenbox.get_X_from_config(clap_config)
        index_combination_size,cost = self.black_box.run_tupels(X, binary)
        self.black_box.recover_workload()
        return index_combination_size,cost
    
class FrayParameterFilter(SelectionAlgorithm):
    def __init__(self, black_box, gray = True, prf = True, random = False):
        SelectionAlgorithm.__init__(self,black_box)
        self.gray = gray
        self.prf = prf
        self.random = random
        self.randomopenbox = ClapboardOpenbox(black_box = self.black_box,random = True, max_runs = 1000)
        self.randomopenbox.run()
        important_parameters = self.randomopenbox.get_important_parameters(rate=0.8)
        logging.info(f"important_parameters is : {important_parameters}")
        best_random_configs = self.randomopenbox.get_optimal_configs()[0]
        logging.info(f"best_random_configs is : {best_random_configs}")
        self.clapboardopenbox = ClapboardOpenbox(black_box = self.black_box, gray = self.gray, prf = self.prf, random = self.random,
                                                 important_parameters = important_parameters, best_random_configs = best_random_configs)

    

    def get_optimal_value(self):
        self.num_pages = self.black_box.get_max_partition_number()
        self.clapboardopenbox.run()
        clap_config = self.clapboardopenbox.get_optimal_configs()[0]
        X, binary = self.clapboardopenbox.get_X_from_config(clap_config)
        index_combination_size,cost = self.black_box.run_tupels(X, binary)
        return index_combination_size,cost    

class AllIndexes(SelectionAlgorithm):
    def __init__(self, black_box):
        SelectionAlgorithm.__init__(self,black_box)
        
    def get_optimal_value(self):
        self.num_pages = self.black_box.get_max_partition_number()
        self.config_space = 2 ** self.black_box.get_attribute_number()  # 默认不会溢出
        X = []
        X.append([(0, self.num_pages)])
        X += [self.config_space - 1]*self.num_pages
        index_combination_size,cost = self.black_box.run_tupels(X, 0)
        return index_combination_size,cost
    
class NoIndexes(SelectionAlgorithm):
    def __init__(self, black_box):
        SelectionAlgorithm.__init__(self,black_box)
        
    def get_optimal_value(self):
        self.num_pages = self.black_box.get_max_partition_number()
        self.config_space = 2 ** self.black_box.get_attribute_number()  # 默认不会溢出
        X = []
        X.append([(0, self.num_pages)])
        X += [0]*self.num_pages
        index_combination_size,cost = self.black_box.run_tupels(X, 0)
        return index_combination_size,cost
    
class AllIndexesPartition(SelectionAlgorithm):
    def __init__(self, black_box):
        SelectionAlgorithm.__init__(self,black_box)
        
    def get_optimal_value(self):
        self.num_pages = self.black_box.get_max_partition_number()
        self.config_space = 2 ** self.black_box.get_attribute_number()  # 默认不会溢出
        X = []
        X.append(self.code_to_partition_tuples((2**self.num_pages)-1, self.num_pages))
        X += [self.config_space - 1]*self.num_pages
        index_combination_size,cost = self.black_box.run_tupels(X, (2**self.num_pages)-1)
        return index_combination_size,cost
    
class NoIndexesPartition(SelectionAlgorithm):
    def __init__(self, black_box):
        SelectionAlgorithm.__init__(self,black_box)
        
    def get_optimal_value(self):
        self.num_pages = self.black_box.get_max_partition_number()
        self.config_space = 2 ** self.black_box.get_attribute_number()  # 默认不会溢出
        X = []
        X.append(self.code_to_partition_tuples((2**self.num_pages)-1, self.num_pages))
        X += [0]*self.num_pages
        index_combination_size,cost = self.black_box.run_tupels(X, (2**self.num_pages)-1)
        return index_combination_size,cost
    