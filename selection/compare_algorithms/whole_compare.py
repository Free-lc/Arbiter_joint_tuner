import logging
from selection.algorithms.casper_partition import Casper
from selection.algorithms.all_partition import AllPartition
from BO.slalom_tree import PartitionTree
import time
PARTITIONALGORITHMS = {
    # "Slalom":PartitionTree,
    # "Casper":Casper,
    "ALL":AllPartition,
}
# INDEXALGORITHMS = [ "Noindex", "Relaxation","Slalom", "Extend"]
# INDEXALGORITHMS = [ "Noindex", "Slalom", "Relaxation"]
INDEXALGORITHMS = [ "Noindex"]
class WholeCompare():
    def __init__(self, black_box) -> None:
        logging.getLogger().setLevel(logging.DEBUG)
        self.black_box = black_box
        self.partition_algs = {}
    def get_convergence_curve(self):
        self.num_pages = self.black_box.get_max_partition_number()
        for name,partition_alg in PARTITIONALGORITHMS.items():
            time_begin = time.time()
            alg_instant = partition_alg(num_pages = self.num_pages,black_box = self.black_box)
            partition_tuples = alg_instant.get_partitions()
            # partition_tuples = [(0, 5),(5, 10)]
            time_end = time.time()
            self.partition_algs[name] = [partition_tuples, time_end-time_begin]
        for name,[partition_tuples,partiton_time] in self.partition_algs.items():
            for index_alg in INDEXALGORITHMS:
                time_begin = time.time()
                if index_alg == "Noindex":
                    index_combination_size,cost = self.black_box.run_extend(partition_tuples, 'noindex')
                elif index_alg == "Extend":
                    index_combination_size,cost = self.black_box.run_extend(partition_tuples)
                elif index_alg == "Autoadmin":
                    index_combination_size,cost = self.black_box.run_extend(partition_tuples, 'autoadmin')
                elif index_alg == "Slalom":
                    index_combination_size,cost = self.black_box.run_slalom(partition_tuples)
                elif index_alg == "Relaxation":
                    index_combination_size,cost = self.black_box.run_extend(partition_tuples, 'relaxation')
                else:
                    assert 0, f"you need to define {index_alg} !!!"
                time_end = time.time()
                logging.info(f"Partition algorithm : {name}, partitioning time :{partiton_time}s, index algorithm : {index_alg}, index_size :{index_combination_size}, cost : {cost}, time : {time_end-time_begin}")
        
        
        