import time
import math
import random
from operator import itemgetter
import numpy as np

from evalution import *

# 对于每棵新树,只将比旧树更优秀的新树添加到森林中,舍弃劣质树
# 上面这句话明显有问题，虽然新树可能没有旧树优秀，但是不能保证新树随机产生的子树没有旧树优秀


class Forest (object):
    def __init__(self, eval_function, method, file_path, max_iterations,
                 area_limit, max_life_time, life_time_limit, transfer_rate):
        """
        This function gets the evaluation function, the ranges of the variables,
        the dimension, the maximum number of iterations, area_limit parameter,
        Life_time parameter and Transfer_rate as input and forms the initial
        Forest. If the input parameters are not provided, the default values in
        the Main function will be used

        Parameters
        ----------
        eval_function: EvaluationFunction
            Evaluation function handler
        method: ValidationMethod
            validation method
        file_path: str
            file path to the dataset
        dim: int
            Dimension of the problem domain
        max_iterations: int
            The predefined maximum number of iterations used for stop condition
        area_limit: int
            The limitation of the forest
        max_life_time: int
            The maximum allowed Age of a tree
        transfer_rate: float
            The percentage of candidate population for global seeding
        """

        self.area_limit = area_limit                 # The limitation of the forest
        # The maximum allowed age of a tree
        self.max_life_time = max_life_time
        # The maximum allowed age of a tree in global_seeding
        self.life_time_limit = life_time_limit
        # The percentage of candidate population
        self.transfer_rate = transfer_rate
        self.max_iterations = max_iterations         # Maximum number of iterations
        self.file_path = file_path
        self.dataset = np.genfromtxt(file_path, delimiter=',')
        # The dimension of the problem domain
        self.dimension = self.dataset.shape[1] - 1
        self.method = method
        self.eval_function = eval_function
        self.past_best_trees = []

        if self.dimension < 5:
            self.LSC = 1  # Local seeding changes (1/5 of the dimension)
            self.GSC = 1  # Global seeding changes
        else:
            # 20 percent (not optimal) of the dimension used in local seeding
            self.LSC = math.floor((2 * self.dimension) / 10)
            # 10 percent (not optimal) of the dimension used in global seeding
            self.GSC = math.floor((1 * self.dimension) / 10)

        self.forest = []
        self.candidate_population = []
        self.best_tree = []
        self.id = 0

        self.tree_groups = []
        '''
        record = [feature_index, count, accuracy_sum, dimension_reduction_sum, accuracy_average, dimension_reduction_average]
        tree_groups_record = [[record, record], []]
        '''
        self.tree_groups_record = []

    def _initialize_forest(self):
        '''
        Invovled variables:
        tree: list
                Age | x | x | x | x | x | x | fitness    | dimension reduction | id | ansters
                x is either 0 or 1
            index:
                0   | 1 | 2 | . | . | . |dim| dimension+1|    dimension+2      | +3 |

        self.best_tree: list
        '''

        init_num = 0

        if self.dimension < 5:
            init_num = 1    #
        else:
            # 20 percent (not optimal) of the dimension used in initial
            init_num = math.floor((2 * self.dimension) / 10)

        high_feature = int(self.dimension * 3 / 4) * [1] + \
            (self.dimension - int(self.dimension * 3 / 4)) * [0]    # 3/4
        mid_feature = int(self.dimension * 1 / 2) * [1] + \
            (self.dimension - int(self.dimension * 1 / 2)) * [0]    # 1/2
        low_feature = int(self.dimension * 1 / 4) * [1] + \
            (self.dimension - int(self.dimension * 1 / 4)) * [0]    # 1/4

        self.__init_ancestor(high_feature, init_num)
        self.__init_ancestor(mid_feature, init_num)
        self.__init_ancestor(low_feature, init_num)

        self.best_tree = self.forest[0][:]

    def __init_ancestor(self, feature_subset, num):
        for i in range(num):
            random.shuffle(feature_subset)
            tree = [0] + feature_subset  # Set age to 0
            fitness = self._get_accuracy(
                tree[1: self.dimension + 1])  # 计算每个 tree 的 fitness
            dimension_reduction = self._get_dimension_reduction_rate(
                tree[1: self.dimension + 1])  # 计算每个 tree 的降维率
            tree.append(fitness)              # 加入准确率
            tree.append(dimension_reduction)  # 加入降维
            tree.append(self.id)              # 加入 id
            tree.append(self.id)              # 加入父子关系
            self.id += 1
            self.forest.append(tree)
            self.tree_groups.append([tree])   # 加入初始族群

            record_list = []
            for i in range(self.dimension):
                record = [i] + [0] * 5        # 每个 record 有 6 个属性
                record_list.append(record)

            self.tree_groups_record.append(record_list)  # 有几个 group 就有几个 record_list
            self.__add_tree_into_group_record(tree)

    def __add_tree_into_group_record(self, tree):
        father_tree_id = tree[self.dimension + 4]
        group_record_list = self.tree_groups_record[father_tree_id]

        # 将 group record 按 feature 排序
        # group record 里有 self.dimension 个 record
        if group_record_list[0] is not None:
            group_record_list = sorted(group_record_list, key=itemgetter(0))

        tree_info = tree[1:self.dimension+3]

        for feature_index in range(self.dimension):
            if feature_index == 1:
                group_record_list[feature_index][1] += 1  # count ++
                group_record_list[feature_index][2] += tree_info[self.dimension]    # add accuracy
                group_record_list[feature_index][3] += tree_info[self.dimension+1]  # add dimension_reduction
                group_record_list[feature_index][4] = group_record_list[feature_index][2] / group_record_list[feature_index][1]  # accuracy average
                group_record_list[feature_index][5] = group_record_list[feature_index][3] / group_record_list[feature_index][1]  # dimension_reduction average

    def __get_group_statistics(self, group):
        # 获得族群统计信息
        record_list = [None] * (self.dimension + 2)
        for i in range(1, self.dimension + 3):
            count = 0
            accuracy_sum = 0
            dimension_reduction_sum = 0
            for j in range(len(group)):
                if group[j][i] == 1:
                    count += 1
                    accuracy_sum += group[j][self.dimension + 1]
                    dimension_reduction_sum += group[j][self.dimension + 1]

            accuracy_average = 0
            dimension_reduction_average = 0

            if count != 0:
                accuracy_average = accuracy_sum/count
                dimension_reduction_average = dimension_reduction_sum/count

            record_list[i-1] = [i-1, count, accuracy_sum, dimension_reduction_sum, accuracy_average, dimension_reduction_average]
        return record_list

    def _local_seeding(self):
        new_trees = []
        for tree in self.forest:
            if tree[0] == 0:       # Perform local seeding on trees with Age 0
##                group = self.tree_groups[tree[self.dimension + 4]]
##                record_list = self.__get_group_statistics(group)
                record_list = self.tree_groups_record[tree[self.dimension + 4]]

                record_list = sorted(
                    record_list,
                    key=itemgetter(4, 5),
                    reverse=True)
                index_according_to_accuracy = [row[0] for row in record_list]
                best_n_to_remain = self.best_tree.count(1)
                cols_to_use = index_according_to_accuracy[best_n_to_remain:]   # some best index remain unchanged
                # Randomly choose LSC variables of the selected tree
                if tree[self.dimension+1] >= 0.85 * self.best_tree[self.dimension+1]:  # good trees
                    new_trees.extend(self.__plant_trees(tree, cols_to_use, 2 * self.LSC))
                else:
                    new_trees.extend(self.__plant_trees(tree, cols_to_use, self.LSC))

            # Increase the Age of all trees except new generated ones in the local
            # seeding stage by 1
            tree[0] = tree[0] + 1

        self.forest.extend(new_trees)  # Merge new trees into forest

    def __plant_trees(self, father_tree, cols_to_use, LSC):
        new_trees = []
        if LSC > len(cols_to_use):
            LSC = len(cols_to_use)
        selected_index = random.sample(cols_to_use, LSC)
        for index in selected_index:
            temp_tree = father_tree[:]
            # change from 0 to 1 or vice versa.
            temp_tree[index] = 1 - temp_tree[index]
            fitness = self._get_accuracy(
                temp_tree[1: self.dimension + 1])
            dimension_reduction = self._get_dimension_reduction_rate(
                temp_tree[1: self.dimension + 1])  # 计算每个 tree 的降维率
            temp_tree[self.dimension + 1] = fitness               # 更新准确率
            temp_tree[self.dimension + 2] = dimension_reduction   # 更新降维
            temp_tree[self.dimension + 3] = self.id               # 更新id
            self.id += 1
            temp_tree.append(father_tree[self.dimension + 3])     # 加入父子关系
            new_trees.append(temp_tree)
            self.tree_groups[temp_tree[self.dimension + 4]].append(temp_tree) # 加入族群
            self.__add_tree_into_group_record(temp_tree)

        return new_trees

    def _population_limiting(self):
        for tree in self.forest:       # Trees with “Age” bigger than “life time” parameter
            if tree[0] > self.max_life_time:
                self.forest.remove(tree)
            if tree[0] > self.life_time_limit:
                self.candidate_population.append(tree)
                if self.forest.count(tree) > 0:
                    self.forest.remove(tree)

        for tree in self.candidate_population:
            if tree[0] > self.max_life_time:
                self.candidate_population.remove(tree)
            else:
                tree[0] += 1

        # The extra trees that exceed “area limit” parameter after sorting the
        # trees according to their fitness value will be dropped
        if len(self.forest) > self.area_limit:
            # sort the forest according to the fitness from high to low
            self.forest = sorted(
                self.forest,
                key=itemgetter(self.dimension + 1, self.dimension + 2),
                reverse=True)
            # 0~area_limit-1 : total area_limit
            self.forest = self.forest[:self.area_limit]

    def _global_seeding(self):
        # 有多少颗树进行 global seeding
        selected_tree_num = int(self.transfer_rate *
                                len(self.candidate_population))

        if selected_tree_num != 0:
            selected_trees_index = random.sample(
                range(len(self.candidate_population)), selected_tree_num)

            self._global_seeding_trees(self.candidate_population,
                                       selected_trees_index)

    def _global_seeding_trees(self, selected_trees, selected_trees_index):
        """ Global seeding 是对 candidate population 中 tranfer rate 个树，每颗树改变 GSC 个参数
            
        """
        for index in selected_trees_index:
            temp_tree = selected_trees[index][:]

            GSC = self.GSC
            if temp_tree[self.dimension + 1] < self.best_tree[self.dimension + 1] * 0.75:
                self.candidate_population.remove(temp_tree)
                GSC = 2 * self.GSC

            selected_variables_index = random.sample(
                range(1, self.dimension + 1), GSC)
            for i in selected_variables_index:
                # The value of each selected variable will be negated
                # (changing from 0 to 1 or vice versa)
                temp_tree[i] = 1 - temp_tree[i]
                fitness = self._get_accuracy(
                    temp_tree[1: self.dimension + 1])
                dimension_reduction = self._get_dimension_reduction_rate(
                    temp_tree[1: self.dimension + 1])  # 计算每个 tree 的降维率
                temp_tree[self.dimension + 1] = fitness
                temp_tree[self.dimension + 2] = dimension_reduction
                temp_tree[self.dimension + 3] = self.id               # 更新id
                self.id += 1
                temp_tree.append(selected_trees[index][self.dimension + 3])        # 加入父子关系
                self.forest.append(temp_tree)
                self.tree_groups[temp_tree[self.dimension + 4]].append(temp_tree)  # 加入族群
                self.__add_tree_into_group_record(temp_tree)

    def _update_best_tree(self):
        # sort the forest according to the fitness from high to low
        self.forest = sorted(self.forest,
                             key=itemgetter(self.dimension + 1, self.dimension + 2),
                             reverse=True)
        if (self.forest[0][self.dimension + 1] > self.best_tree[self.dimension + 1]) or \
           ((self.forest[0][self.dimension + 1] == self.best_tree[self.dimension + 1]) and
            (self.forest[0][self.dimension + 2] > self.best_tree[self.dimension + 2])):
            # 如果森林里最好的树比记录的最好的树优，更新记录
            # 或者，准确度相同情况下，森林里最好的树降维比率更大
            self.forest[0][0] = 0   # set best tree's age to 0
            self.best_tree = self.forest[0][:]
        else:
            # 此时，森林里最好的树应该就是过去记录的某颗最好的树
            self.best_tree[0] = 0
            self.forest[0] = self.best_tree[:]
            print("-------self.best_tree-----------")
            print(self.best_tree)

    def __invisible_hand(self):
        self.past_best_trees.append(self.best_tree)

        for tree in self.forest:
            # 如果是和 best tree 一个族群的
            if tree[self.dimension + 3] == self.best_tree[self.dimension + 3] and tree[0] != 0:
                self.forest.remove(tree)

        for tree in self.candidate_population:
            if tree[self.dimension + 3] == self.best_tree[self.dimension + 3] and tree[0] != 0:
                self.candidate_population.remove(tree)

        self.best_tree = self.forest[0]

    def _get_accuracy(self, feature_subset):
        '''
        Returns
        ------------------------
        fitness: float
            classification accuracy
        '''

        fitness = Evaluation.get_accuracy(
            self.eval_function, feature_subset,
            self.dataset, self.method)

# print("-------feature_subset---------")
# print(feature_subset)
# print("-------fitness-------")
# print(fitness)

        return fitness

    def _get_dimension_reduction_rate(self, feature_subset):
        return Evaluation.get_dimension_reduction_rate(feature_subset)

    def evolution(self):
        '''
        Returns
        -----------------------
        best_tree: list
            best tree
        '''
        self._initialize_forest()

        for i in range(self.max_iterations):
            print("iteration round "+ str(i))
            self._local_seeding()
            self._population_limiting()
            self._global_seeding()
            self._update_best_tree()
            if i > 15 and (i-15) % 20 == 0:
                self.__invisible_hand()

        self.past_best_trees.append(self.best_tree)

        self.past_best_trees = sorted(self.past_best_trees,
                                      key=itemgetter(self.dimension + 1, self.dimension + 2),
                                      reverse=True)

        return self.past_best_trees[0]

if __name__ == '__main__':
    start_time = time.time()
    #    file_path = r"C:\Users\Administrator\Desktop\11111\dataset\low\heart.csv"
    file_path = r".\dataset\high\srbct.csv"
    forest = Forest(EvaluationFunction.ONE_NN, ValidationMethod.SEVEN_THREE,
                    file_path, 100, 50, 20, 15, 0.05)
    print(forest.evolution())
    print("--- %s seconds ---" % (time.time() - start_time))
