from typing import List, Tuple, Optional
import math
from openbox import space as sp, Optimizer
import numpy as np
import sys
sys.path.append("/mydata/BO/index_selection_evaluation")
from index_selection_evaluation.selection.db_openbox import BlackBox

class TreeNode():
    def __init__(
        self,
        depth: int, # 似乎暂时用不上
        partition_range: Tuple[int], # 注意此处partition range是以块大小为单位的，左闭右开区间，如[0, 2]表示0、1两个物理页
        max_performance: float,
        parent: Optional['TreeNode'] = None,
        l_child: Optional['TreeNode'] = None,
        r_child: Optional['TreeNode'] = None,
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
        self.depth = depth
        self.partition_range = partition_range
        self.max_performance = max_performance
        self.parent = parent
        self.l_child = l_child
        self.r_child = r_child
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


    def split_partition(self):

        split_pos = sum(self.partition_range) // 2
        left_range = (self.partition_range[0], split_pos)
        right_range = (split_pos, self.partition_range[1])

        left_child = TreeNode(
            depth = self.depth + 1,
            partition_range = left_range,
            max_performance = 0,
            parent = self
        )

        right_child = TreeNode(
            depth = self.depth + 1,
            partition_range = right_range,
            max_performance = 0,
            parent = self
        )


        self.add_left_child(left_child)
        self.add_right_child(right_child)

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
        depth: int = 0
    ):
        self.depth = depth
        self.num_pages = num_pages
        self.root = TreeNode(
            depth = 0,
            partition_range = (0, self.num_pages),
            max_performance = 0,
            parent = None,
        )
        self.current_node = self.root
        self.partitions = []
        self.partition_state = 'l'
        self.indexs = []
        self.min_cost = 0

        # ? added
        self.black_box = BlackBox()
        self.num_attr = self.black_box.get_attribute_number()

        self.db_config = {
            'num_indices': self.num_attr, #索引种类数
            'budget': 600
            
        }

        param = {'x': (0, 2**self.num_attr, 0)}
        self.space = sp.Space()
        self.space.add_variables([
            sp.Int(name, *para) for name, para in param.items()
        ])

        self.opt = Optimizer(
            objective_function = self.max_performance_within_partition,
            config_space = self.space,
            num_constraints = 0,
            num_objectives = 1,
            surrogate_type = 'gp',
            acq_optimizer_type = 'random_scipy',
            max_runs = 50,
            task_id = 'basic_search',
            logging_dir = 'logs_basic_search',
            visualization = 'basic'
        )

    # def split_partition(self):
    #     split_pos = sum(self.current_node.partition_range) // 2
    #     left_range = (self.current_node.partition_range[0], split_pos)
    #     right_range = (split_pos, self.current_node.partition_range[1])

    #     left_child = TreeNode(
    #         depth = self.current_node.depth + 1,
    #         partition_range = left_range,
    #         max_performance = 0,
    #         parent = self.current_node,
    #     )

    #     right_child = TreeNode(
    #         depth = self.current_node.depth + 1,
    #         partition_range = right_range,
    #         max_performance = 0,
    #         parent = self.current_node,
    #     )


    #     self.current_node.add_left_child(left_child)
    #     self.current_node.add_right_child(right_child)

    def build_best_performance_tree(self):  # 先序遍历创建二叉树，并且直接把叶子结点取出来作为列表、

        self.root.calculate_best_performance()

        self.current_node = self.root

        while self.root.is_best is False:
            if self.current_node.partition_range[1] - self.current_node.partition_range[0] <= 1:    # 粒度达到单个物理页，无法再分，应当回溯
                self.current_node.is_best = True
                self.current_node = self.current_node.parent

            elif self.current_node.l_child is None:   # 还没有进一步划分，说明还在向下分区过程中

                self.current_node.split_partition() # 将当前分区划分成两个子分区

                self.partition_state = 'l'
                self.calculate_best_performance()
                self.partition_state = 'r'
                self.calculate_best_performance()  # 计算两个分区的最佳表现
                self.partition_state = 'c'

                l_max_performance = self.current_node.l_child.max_performance  
                r_max_performance = self.current_node.r_child.max_performance 
                if self.current_node.max_performance >= l_max_performance + r_max_performance:  # ? 是否相加待定
                    self.current_node.is_best = True    # 父分区的性能比划分后要好，则认为划分无法带来更大的收益，当前这种划分达到了stable state
                    self.current_node.l_child = self.current_node.r_child = None    # 删除左右子树
                    self.partitions.append(self.current_node.partition_range)   # 将当前分区的范围加入到分区列表中
                    self.current_node = self.current_node.parent    # 回溯

                else:
                    self.current_node = self.current_node.l_child   # 对左子树求最佳performance

            elif self.current_node.l_child.is_best == True: # 说明已经遍历过左子树，左孩子已达最优
                if self.current_node.r_child.is_best == True:   # 接下来判断是否遍历了右子树，如果遍历了右子树，那么这个时候需要回溯
                    self.current_node.max_performance = self.current_node.l_child.max_performance + self.current_node.r_child.max_performance   # ? 是否相加待定,
                    # 注意到左右孩子的最佳表现之和大于父节点是显然的，否则不会创建左右子树
                    self.current_node.is_best = True
                    self.current_node = self.current_node.parent
                
                else:
                    self.current_node = self.current_node.r_child

    def max_performance_within_partition(self, config: sp.Configuration):

        X = np.array([config['x']])

        pr = ()
        id = len(self.partitions)

        if self.partition_state == 'c':
            pr = self.current_node.partition_range

        elif self.partition_state == 'l':
            pr = self.current_node.l_child.partition_range

        else:
            pr = self.current_node.r_child.partition_range
            id += 1

        index_combination_size, cost, no_index_cost = self.black_box.run_tuple(
            id,   # 当前处理的是第几个逻辑分区
            pr,  # 当前处理的分区范围
            X
        )

        obj = (no_index_cost - cost) / min(index_combination_size, 1e-5) 

        result = {
            'objectives': [obj, ],
            'constraints': [index_combination_size - self.db_config['budget'], ]
        } 
        return result

    def max_performance_global(self, config: sp.Configuration):
        X = np.array([config[f'x{i}'] for i in range(len(self.partitions))])
        index_combination_size, cost = self.black_box.run_final(self.partitions, X)
        result = {
            'objectives': [cost,],
            'constraints': [index_combination_size,]
        }
        return result


    def calculate_best_performance(self):
        """
        计算当前分区中的最佳索引性能，并赋值给self.max_performance
        也就是在当前空间中搜索
        """
        #raise NotImplementedError # TODO: 计算当前分区中的最佳索引性能，并赋值给

        history = self.opt.run()
        result = history.get_incumbent_value()
        if self.partition_state == 'c':
            self.current_node.max_performance = result

        elif self.partition_state == 'l':
            self.current_node.l_child.max_performance = result

        else:
            self.current_node.r_child.max_performance = result

        

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
        
    def global_search(self):
        params = {f'x{i}': (0, 2**self.num_attr, 0) for i in range(len(self.partitions))}
        self.space = sp.Space()
        self.space.add_variables([
            sp.Int(name, *para) for name, para in params.items()
        ])
        opt = Optimizer(
            objective_function = self.max_performance_global,
            config_space = self.space,
            num_constraints = 1,
            num_objectives = 1,
            surrogate_type = 'gp',
            acq_optimizer_type = 'random_scipy',
            max_runs = 50,
            task_id = 'basic_search',
            logging_dir = 'logs_basic_search',
            visualization = 'basic'
        )

        history = opt.run()
        self.indexs = history.get_incumbent_configs
        self.min_cost= history.get_incumbent_value


if __name__ == "__main__":
    partition_tree = PartitionTree(num_pages = 20)
    partition_tree.build_best_performance_tree()
    partition_tree.global_search()
    partitions = partition_tree.get_partitions()
    print(partitions)

