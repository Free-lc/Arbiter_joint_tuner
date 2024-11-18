from selection.compare_algorithms.recommend_algorithm import SelectionAlgorithm
from BO.slalom_tree import PartitionTree
class Slalom(SelectionAlgorithm):
    def __init__(self, black_box):
        SelectionAlgorithm.__init__(self,black_box)
        
    def get_optimal_value(self):
        self.num_pages = self.black_box.get_max_partition_number()
        partition_tree = PartitionTree(num_pages = self.num_pages,black_box = self.black_box)
        partition_tuples = partition_tree.get_partitions()
        index_combination_size,cost = self.black_box.run_slalom(partition_tuples)
        return index_combination_size,cost