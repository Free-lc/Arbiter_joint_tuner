from selection.compare_algorithms.recommend_algorithm import SelectionAlgorithm
from selection.algorithms.casper_partition import Casper

class CasperExtend(SelectionAlgorithm):
    def __init__(self, black_box):
        SelectionAlgorithm.__init__(self,black_box)
        
    def get_optimal_value(self):
        self.num_pages = self.black_box.get_max_partition_number()
        partition_alg = Casper(self.num_pages, self.black_box)
        partition_tuples = partition_alg.get_partitions()
        index_combination_size,cost = self.black_box.run_extend(partition_tuples)
        return index_combination_size,cost