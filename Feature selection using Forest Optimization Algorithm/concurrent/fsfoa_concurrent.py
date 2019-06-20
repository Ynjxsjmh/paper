import time
import math
import random
from operator import itemgetter
import numpy as np
# import .evaluation
from enum import Enum, unique
from sklearn import svm
from sklearn import tree as Tree
# K最近邻(kNN，k-NearestNeighbor)分类算法
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score   # K折交叉验证模块
from sklearn.model_selection import train_test_split  # 分割数据模块

import concurrent.futures


class Forest (object):
    def __init__(self, eval_function, file_path, max_iterations,
                 area_limit, max_life_time, transfer_rate):
        """
        This function gets the evaluation function, the ranges of the variables,
        the dimension, the maximum number of iterations, area_limit parameter,
        Life_time parameter and Transfer_rate as input and forms the initial
        Forest. If the input parameters are not provided, the default values in
        the Main function will be used

        Parameters
        ----------
        eval_function:
            Evaluation function handler
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
        # The percentage of candidate population
        self.transfer_rate = transfer_rate
        self.max_iterations = max_iterations         # Maximum number of iterations
        self.file_path = file_path
        self.dataset = np.genfromtxt(file_path, delimiter=',')
        # The dimension of the problem domain
        self.dimension = self.dataset.shape[1] - 1

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

        self._initialize_forest()

    def _initialize_forest(self):
        '''
        Invovled variables:
        tree: list
                Age | x | x | x | x | x | x | fitness
                x is either 0 or 1
            index:
                0   | 1 | 2 | . | . | . |dim| dimension+1

        self.best_tree: list
        '''
        init_num = random.randint(
            1, self.dimension + 1)  # Randomly generated tree num
        '''上面有点问题'''

        for i in range(init_num):
            # Randomly generated dimension+1 tree variables with value between
            # 0 and 1
            tree = [random.randint(0, 1) for _ in range(self.dimension + 1)]
            tree[0] = 0            # Set age to 0
            fitness = self._accuracy_calculator(
                tree[1: self.dimension + 1])  # 计算每个 tree 的 fitness
            tree.append(fitness)
            self.forest.append(tree)

        self.best_tree = self.forest[0][:]

    def _accuracy_calculator(self, feature_subset):
        '''
        Returns
        ------------------------
        fitness: float
            classification accuracy
        '''

        fitness = Evaluation.k_nearest_neighbor(
            1, feature_subset, self.dataset, ValidationMethod.SEVEN_THREE)
    #        fitness = Evaluation.svm_rbf(feature_subset, f.dataset, ValidationMethod.SEVEN_THREE)

    # print("-------feature_subset---------")
    # print(feature_subset)
    # print("-------fitness-------")
    # print(fitness)

        return fitness


def _local_seeding(f, is_accelerate):
    start_time = time.time()

    if is_accelerate:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(__local_seeding_helper, f, tree) for tree in f.forest]
            for future in concurrent.futures.as_completed(futures):
                # Merge new trees into forest
                f.forest.extend(future.result()) # Merge new trees into forest
    else:
        new_trees = []
        for tree in f.forest:
            if tree[0] == 0:       # Perform local seeding on trees with Age 0
                # Randomly choose LSC variables of the selected tree
                selected_index = random.sample(
                    range(1, f.dimension + 1), f.LSC)
                for index in selected_index:
                    temp_tree = tree[:]
                    # change from 0 to 1 or vice versa.
                    temp_tree[index] = 1 - temp_tree[index]
                    fitness = f._accuracy_calculator(
                        temp_tree[1: f.dimension + 1])
                    temp_tree[f.dimension + 1] = fitness
                    new_trees.append(temp_tree)

            # Increase the Age of all trees new generated ones in the local
            # seeding stage by 1
            tree[0] = tree[0] + 1

        f.forest.extend(new_trees)  # Merge new trees into forest

    print("--- %s seconds ---" % (time.time() - start_time))


def __local_seeding_helper(f, tree):
    new_trees = []

    if tree[0] == 0:       # Perform local seeding on trees with Age 0
        # Randomly choose LSC variables of the selected tree
        selected_index = random.sample(
            range(1, f.dimension + 1), f.LSC)
        for index in selected_index:
            temp_tree = tree[:]
            # change from 0 to 1 or vice versa.
            temp_tree[index] = 1 - temp_tree[index]
            fitness = f._accuracy_calculator(temp_tree[1: f.dimension + 1])
            temp_tree[f.dimension + 1] = fitness
            new_trees.append(temp_tree)

    # Increase the Age of all trees new generated ones in the local
    # seeding stage by 1
    tree[0] = tree[0] + 1

    return new_trees


def _population_limiting(f, is_accelerate):
    if (is_accelerate):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for tree in f.forest:
                executor.submit(_population_limiting_helper, f, tree)
    else:
        for tree in f.forest:                 # Trees with “Age” bigger than “life time” parameter
            if tree[0] > f.max_life_time:
                f.candidate_population.append(tree)
                f.forest.remove(tree)

    # The extra trees that exceed “area limit” parameter after sorting the
    # trees according to their fitness value will be dropped
    if len(f.forest) > f.area_limit:
        # sort the forest according to the fitness from high to low
        f.forest = sorted(
            f.forest, key=itemgetter(
                f.dimension + 1), reverse=True)
        # 0~area_limit-1 : total area_limit
        f.forest = f.forest[:f.area_limit]


def _population_limiting_helper(f, tree):
    if tree[0] > f.max_life_time:
        f.candidate_population.append(tree)
        f.forest.remove(tree)


def _global_seeding(f, is_accelerate):
    # 有多少颗树进行 global seeding
    selected_tree_num = int(f.transfer_rate *
                            len(f.candidate_population))

    if selected_tree_num != 0:
        selected_trees_index = random.sample(
            range(len(f.candidate_population)), selected_tree_num)

        if is_accelerate:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                for index in selected_trees_index:
                    executor.submit(_population_limiting_helper, f, index)
        else:
            for index in selected_trees_index:
                temp_tree = f.candidate_population[index][:]
                selected_variables_index = random.sample(
                    range(1, f.dimension + 1), f.GSC)
                for i in selected_variables_index:
                    # The value of each selected variable will be negated
                    # (changing from 0 to 1 or vice versa)
                    temp_tree[i] = 1 - temp_tree[i]
                fitness = f._accuracy_calculator(
                    temp_tree[1: f.dimension + 1])
                temp_tree[f.dimension + 1] = fitness
                f.forest.append(temp_tree)


def _global_seeding_helper(f, index):
    temp_tree = f.candidate_population[index][:]
    selected_variables_index = random.sample(
        range(1, f.dimension + 1), f.GSC)
    for i in selected_variables_index:
        # The value of each selected variable will be negated
        # (changing from 0 to 1 or vice versa)
        temp_tree[i] = 1 - temp_tree[i]
    fitness = f._accuracy_calculator(
        temp_tree[1: f.dimension + 1])
    temp_tree[f.dimension + 1] = fitness
    f.forest.append(temp_tree)


def _update_best_tree(f):
    # sort the forest according to the fitness from high to low
    f.forest = sorted(
        f.forest, key=itemgetter(
            f.dimension + 1), reverse=True)
    if f.forest[0][f.dimension +
                   1] > f.best_tree[f.dimension + 1]:
        # 如果森林里最好的树比记录的最好的树优，更新记录
        f.forest[0][0] = 0   # set best tree's age to 0
        f.best_tree = f.forest[0][:]
    else:
        # 此时，森林里最好的树应该就是过去记录的某颗最好的树
        f.best_tree[0] = 0
        f.forest[0] = f.best_tree[:]
    print("-------f.best_tree-----------")
    print(f.best_tree)


def evolution(f):
    '''
    Returns
    -----------------------
    best_tree: list
        best tree
    '''

    for i in range(f.max_iterations):
        if (len(f.forest) > 1000):
            _local_seeding(f, True)
        else:
            _local_seeding(f, False)

        if (len(f.forest) > 1000):
            _population_limiting(f, True)
        else:
            _population_limiting(f, False)

        if (len(f.candidate_population)*0.05 > 1000):
            _global_seeding(f, True)
        else:
            _global_seeding(f, False)

        _update_best_tree(f)

    return f.best_tree


# ------------------------------------------------
@unique               # 该装饰器可以帮助我们检查保证没有重复值
class ValidationMethod(Enum):
    TEN_FOLD = 1      # 10-fold
    TWO_FOLD = 2      # 2-fold
    SEVEN_THREE = 3   # 70%-30%


class Evaluation(object):
    @staticmethod
    def k_nearest_neighbor(k, feature_subset, dataset, method):
        """
        Parameters
        ----------
        k: int
            n_neighbors
        feature_subset: list
            one possible feature subset
        dataset: ndarray
            target dataset
        method: ValidationMethod
            validation method

        Returns
        -------
        fitness: float
            classification accuracy
        """
        X, y = Evaluation._getXy(feature_subset, dataset)

        knn = KNeighborsClassifier(n_neighbors=k)  # 建立模型

        return Evaluation._get_accuracy(knn, X, y, method)

    @staticmethod
    def svm_rbf(feature_subset, dataset, method):
        X, y = Evaluation._getXy(feature_subset, dataset)

        clf = svm.SVC(gamma='scale')

        return Evaluation._get_accuracy(clf, X, y, method)

    @staticmethod
    def cff_cart(feature_subset, dataset, method):
        X, y = Evaluation._getXy(feature_subset, dataset)

        clf = Tree.DecisionTreeRegressor()

        return Evaluation._get_accuracy(clf, X, y, method)

    @staticmethod
    def _getXy(feature_subset, dataset):
        columns = dataset.shape[1]
        cols_to_use = [
            i for i in range(
                len(feature_subset)) if feature_subset[i] == 1]

        X = dataset[:, cols_to_use]
        y = dataset[:, columns - 1]

        return [X, y]

    @staticmethod
    def _get_accuracy(clf, X, y, method):
        '''
        Parameters
        -----------------------
        clf:
           scikit-learn classifier
        X: two dimensional ndarray
           特征矩阵
        y: two dimensional ndarray
           目标向量
        method: ValidationMethod
           validation method

        Returns
        -----------------------
        accuracy: float
           prediction accuracy
        '''
        accuracy = 0.0

        if X.size == 0:
            # if X = []. Caused by cols_to use = [], which means feature_subset
            # is all zero, return 0 accuracy
            return accuracy

        if method == ValidationMethod.SEVEN_THREE:
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42)
            clf.fit(X_train, y_train)  # 训练模型
            accuracy = clf.score(X_test, y_test)

        elif method == ValidationMethod.TEN_FOLD:
            accuracy = cross_val_accuracy(
                clf, X, y, cv=10, scoring='accuracy').mean()

        elif method == ValidationMethod.TWO_FOLD:
            accuracy = cross_val_accuracy(
                clf, X, y, cv=2, scoring='accuracy').mean()

        return accuracy


if __name__ == '__main__':
    start_time = time.time()
    file_path = r"..\dataset\low\ionosphere.csv"
    forest = Forest(1, file_path, 100, 50, 15, 0.05)
    print(evolution(forest))
    print("--- %s seconds ---" % (time.time() - start_time))
