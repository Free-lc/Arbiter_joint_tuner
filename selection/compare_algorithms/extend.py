from selection.compare_algorithms.recommend_algorithm import SelectionAlgorithm

class Extend(SelectionAlgorithm):
    def __init__(self, black_box):
        SelectionAlgorithm.__init__(self,black_box)
        
    def get_optimal_value(self):
        self.num_pages = self.black_box.get_max_partition_number()
        partition_tuples = self.code_to_partition_tuples(2 ** self.num_pages - 1,self.num_pages)
        index_combination_size,cost = self.black_box.run_extend(partition_tuples)
        return index_combination_size,cost