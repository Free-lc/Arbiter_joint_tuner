import sys
import random

class AllPartition():
    def __init__(self, num_pages, black_box) -> None:
        self.num_pages = num_pages
        self.black_box = black_box
        self.db_connector = self.black_box.db_connector
        self.random_read_cost = int(self.db_connector.exec_fetch("SHOW random_page_cost")[0])
        self.sequential_read_cost = int(self.db_connector.exec_fetch("SHOW seq_page_cost")[0])
        random.seed(self.black_box.seed)
    
    def get_partitions(self):
        self.pq = [0] * self.num_pages  # point query
        self.rs = [0] * self.num_pages  # range query start
        self.sc = [0] * self.num_pages  # scans
        self.re = [0] * self.num_pages  # range query end
        p_code = 2 ** (self.num_pages - 1) - 1
        partition_tuples =  self.black_box.code_to_partition_tuples(p_code, self.num_pages)
        return partition_tuples

    
