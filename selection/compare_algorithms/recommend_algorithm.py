import logging
class SelectionAlgorithm:
    def __init__(self, black_box):
        logging.getLogger().setLevel(logging.DEBUG)
        self.black_box = black_box
    def get_optimal_value(self):
        return 0
    def code_to_partition_tuples(self, code: int, dim_configs: int = 8):
        split_points = []
        partition_tuple_list = [(0,dim_configs)]
        pos = 0
        while code > 0:
            digit = code & 1
            if digit == 1:
                split_points.append(dim_configs - 1 - pos)
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