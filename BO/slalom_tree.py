from typing import List, Tuple, Optional
import sys
import math
from openbox import space as sp, Optimizer
import numpy as np

# sys.path.append("/mydata/BO/index_selection_evaluation")
from selection.db_openbox import BlackBox
from collections import deque
from selection.slalom_partition import decide_partition_size

class TreeNode():
    def __init__(
        self,
        partition_range: Tuple[int], # 注意此处partition range是以块大小为单位的，左闭右开区间，如[0, 2]表示0、1两个物理页
        max_performance: float,
        parent: Optional['TreeNode'] = None,
        children: Optional[List['TreeNode']] = None,
        is_best: bool = False
    ):
        """
        构造结点

        参数：
        - depth: 当前结点所在深度，该字段没啥用但是还是得传值因为懒得改了
        - partition_range：List[Tuple]，当前结点表示的分区范围，以物理页为单位，左闭右开区间
        - max_performance：初始化时默认为0
        - parent：父节点
        - l_child：左孩子
        - r_child：右孩子
        - is_best：当前结点的所有子树是否已经遍历完毕，即当前分区是否已经找到了最优的子分区方案
        """
        self.partition_range = partition_range
        self.max_performance = max_performance
        self.parent = parent
        self.children = children if children is not None else []
        self.is_best = is_best
        self.partition_id = 0
        
    def add_left_child(self, l_child: 'TreeNode'):
        if self.l_child is None:
            self.l_child = l_child
            l_child.parent = self

    def add_right_child(self, r_child: 'TreeNode'):
        if self.r_child is None:
            self.r_child = r_child
            r_child.parent = self


    def split_partition(self, num_partitions):
        left, right = self.partition_range
        if num_partitions > right - left:
            num_partitions = right - left
        if num_partitions > 1:
            left, right = self.partition_range
            partitions = np.floor(np.linspace(left, right, num_partitions + 1)).astype(int)
            partition_tuple_list = [(partitions[i], partitions[i+1]) for i in range(len(partitions)-1)]
            for partition_tuple in partition_tuple_list:
                child = TreeNode(
                    partition_range=partition_tuple,
                    max_performance=0,
                    parent=self,
                )
                self.children.append(child)

        return self.children

    def calculate_best_performance(self):
        """
        计算当前分区中的最佳索引性能，并赋值给self.max_performance
        也就是在当前空间中搜索
        """
        pass # TODO: 计算当前分区中的最佳索引性能，并赋值给
    

class PartitionTree():
    def __init__(
        self,
        num_pages: int = 1,
        black_box = None
    ):
        self.num_pages = num_pages
        self.black_box = black_box
        self.db_connector = self.black_box.db_connector
        self.root = TreeNode(
            partition_range = (0, self.num_pages),
            max_performance = 0,
            parent = None,
        )
        self.current_node = self.root
        self.partitions = []
        self.partition_state = 'l'
        self.indexs = []
        self.min_cost = 0
        self.queue = deque()
        self.build_best_performance_tree()

    def build_best_performance_tree(self):  # 先序遍历创建二叉树，并且直接把叶子结点取出来作为列表、
        self.queue.append(self.root)
        while len(self.queue) != 0:
            self.current_node = self.queue.popleft()
            assert self.db_connector, "undefinied db_connector "
            left, right = self.current_node.partition_range
            data, selectivity = self.black_box.get_selectivity(self.current_node.partition_range)
            num_partitions = decide_partition_size(data, selectivity, self.current_node.partition_range)  # * 需要添加该函数
            if num_partitions == 1 or right - left <= 1:
                self.partitions.append(self.current_node.partition_range)
            else:
                children = self.current_node.split_partition(num_partitions)
                for child in children:
                    self.queue.append(child)
        self.partitions.sort(key = lambda x: x[0])

    def predict_num(self):
        if sum(self.current_node.partition_range) % 2 == 0:
            return sum(self.current_node.partition_range) // 2
        else:
            return sum(self.current_node.partition_range)
    

    def max_performance_global(self, config: sp.Configuration):
        X = np.array([config[f'x{i}'] for i in range(len(self.partitions))])
        index_combination_size, cost = self.black_box.run_final(self.partitions, X)
        result = {
            'objectives': [cost,],
            'constraints': [index_combination_size,]
        }
        return result

        

    def get_partitions(self, out_list: bool = True):
        """
        得到最佳分区方案，注意必须要先调用build_best_performance_tree产生最佳的分区树后才可以使用

        参数：
        - out_list: 是否以划分点的形式输出，若为True则输出为[1,3,5]，否则为[(0,1), (1,3), (3,5)]

        """
        if out_list:
            # TODO: 将self.partitions转化成表示切分点的list
            pass

        return self.partitions
        


if __name__ == "__main__":
    # partition_tree = PartitionTree(num_pages = 8)
    # partition_tree.build_best_performance_tree()
    # print(partition_tree.get_partitions())

    # partitions = np.floor(np.linspace(0, 20, 5 + 1)).astype(int)
    # print(partitions)
    print(sys.path)