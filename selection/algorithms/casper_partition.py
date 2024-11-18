import sys
import random

class Casper():
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
        min_cost,min_code = self.analyse_workload()
        partition_tuples =  self.black_box.code_to_partition_tuples(min_code, self.num_pages)
        return partition_tuples

    
    def analyse_workload(self):
        single_queries = self.black_box.single_workload.queries
        for query_id,query in enumerate(single_queries):
            # Collects statistics on data blocks accessed by the query
            query_block_id = []
            for block_id in range(self.num_pages):
                skip = self.black_box._filtering_(self.black_box.workloads[block_id].queries[query_id])
                if skip:
                    continue
                else:
                    query_block_id.append(block_id)
            block_ranges = self.group_consecutive(query_block_id)
            if query.query_class == 'point':
                for block_range in block_ranges:
                    for block_id in range(block_range[0], block_range[1]+1):
                        self.pq[block_id] += query.frequency
            elif query.query_class == 'range':
                for block_range in block_ranges:
                    for block_id in range(block_range[0], block_range[1]+1):
                        if block_id == block_range[0]:
                            self.rs[block_id] += query.frequency
                        elif block_id == block_range[1]:
                            self.re[block_id] += query.frequency
                        else:
                            self.sc[block_id] += query.frequency
        min_cost = sys.maxsize
        min_code = 0
        bck_read = [0] * self.num_pages
        fwd_read = [0] * self.num_pages
        sample_code = random.sample(range(2 ** (self.num_pages - 1)), min(10000, 2 ** (self.num_pages - 1)))
        for iter in sample_code:
            # iter represents the partition code
            num = iter
            code = [0] * (self.num_pages - 1)
            for i in range(self.num_pages - 1):
                bit  = num & 1
                code[i] = bit
                num = num >> 1
            code.reverse()
            cost = 0
            code.append(0)
            for i in range(self.num_pages):
                # get bck_read(i)
                for j in range(0, i):
                    intermediate = 1
                    for k in range(j, i):
                        intermediate *= 1 - code[k]
                    bck_read[i] += intermediate
                # get fwd_read(i)
                for j in range(0, self.num_pages-i):
                    intermediate = 1
                    for k in range(i, self.num_pages-j):
                        intermediate *= (1 - code[k])
                    fwd_read[i] += intermediate
                # get cost_rs(rs_)
                cost_rs = self.rs[i]*self.random_read_cost + self.rs[i]*self.sequential_read_cost*bck_read[i]
                cost_re = self.re[i]*self.sequential_read_cost + self.re[i]*self.sequential_read_cost*fwd_read[i]
                cost_sc = self.sc[i]*self.sequential_read_cost
                range_query_cost = cost_rs + cost_re + cost_sc
                point_query_cost = self.pq[i]*self.random_read_cost*(1+ fwd_read[i] +bck_read[i])
                cost = range_query_cost + point_query_cost
            if cost < min_cost:
                min_cost = cost
                min_code = iter
                
        return min_cost,min_code
    
    def group_consecutive(self, nums):
        if not nums:
            return []

        nums.sort()
        groups = []
        start = nums[0]
        end = nums[0]

        for i in range(1, len(nums)):
            if nums[i] - end == 1:  # If the current number is consecutive to the previous one
                end = nums[i]
            else:
                groups.append([start, end])
                start = nums[i]
                end = nums[i]
        groups.append([start, end])  # Append the last group

        return groups

    
            


        


    