import time
import math
import random
from operator import itemgetter
import numpy as np
# import .evaluation
from enum import Enum, unique
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier   # K最近邻(kNN，k-NearestNeighbor)分类算法
from sklearn.model_selection import cross_val_score  # K折交叉验证模块
from sklearn.model_selection import train_test_split # 分割数据模块

# 对于每棵新树,只将比旧树更优秀的新树添加到森林中,舍弃劣质树
# 上面这句话明显有问题，虽然新树可能没有旧树优秀，但是不能保证新树随机产生的子树没有旧树优秀

class Forest (object):
    def __init__(self, eval_function, file_path, max_iterations, area_limit, max_life_time, transfer_rate):
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
        self.max_life_time = max_life_time           # The maximum allowed age of a tree
        self.transfer_rate = transfer_rate           # The percentage of candidate population
        self.max_iterations = max_iterations         # Maximum number of iterations
        self.file_path = file_path
        self.dataset = np.genfromtxt(file_path, delimiter=',')
        self.dimension = self.dataset.shape[1] - 1   # The dimension of the problem domain

        if self.dimension < 5:
            self.LSC = 1  # Local seeding changes (1/5 of the dimension)
            self.GSC = 1  # Global seeding changes
        else:
            self.LSC = math.floor((2*self.dimension)/10) # 20 percent (not optimal) of the dimension used in local seeding
            self.GSC = math.floor((1*self.dimension)/10) # 10 percent (not optimal) of the dimension used in global seeding

        self.forest = []
        self.candidate_population = []
        self.best_tree = []


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
        init_num = self.area_limit

        forward_num = int(2*init_num/3)        # 森林中 2/3 的树使用 10% 的特征数
        backward_num = init_num - forward_num  # 其他 1/3 使用 K 个特征数(K 是介于 1/2 特征数和特征总数之间的随机数)

        forward_feature = int(self.dimension * 0.1) * [1] + (self.dimension-int(self.dimension*0.1))*[0]
        backward_feature_num = random.randint(int(self.dimension/2), self.dimension+1) # 产生 K 个特征数
        backward_feature = backward_feature_num * [1] + (self.dimension - backward_feature_num)*[0]
        for i in range(forward_num):
            random.shuffle(forward_feature)
            tree = [0] + forward_feature   # Set age to 0
            fitness = self._fitness_calculator(tree[1: self.dimension+1]) #计算每个 tree 的 fitness
            tree.append(fitness)
            self.forest.append(tree)

        for i in range(backward_num):
            random.shuffle(backward_feature)
            tree = [0] + backward_feature  # Set age to 0
            fitness = self._fitness_calculator(tree[1: self.dimension+1]) #计算每个 tree 的 fitness
            tree.append(fitness)
            self.forest.append(tree)

        self.best_tree = self.forest[0][:]


    def _local_seeding(self):
        new_trees = []
        for tree in self.forest:
            if tree[0] == 0:       # Perform local seeding on trees with Age 0
                selected_index = random.sample(range(1, self.dimension+1), self.LSC)  # Randomly choose LSC variables of the selected tree
                temp_tree = tree[:]
                for index in selected_index:
                    temp_tree[index] = 1 - temp_tree[index]   # change from 0 to 1 or vice versa.
                    fitness = self._fitness_calculator(temp_tree[1: self.dimension+1])
                    temp_tree[self.dimension+1] = fitness
                    if temp_tree[self.dimension+1] > tree[self.dimension+1]:
                        new_trees.append(temp_tree)

            tree[0] = tree[0] + 1  # Increase the Age of all trees new generated ones in the local seeding stage by 1

        self.forest.extend(new_trees)  # Merge new trees into forest


    def _population_limiting(self):
        for tree in self.forest:                 # Trees with “Age” bigger than “life time” parameter
            if tree[0] > self.max_life_time:
                self.candidate_population.append(tree)
                self.forest.remove(tree)

        if len(self.forest) > self.area_limit:   # The extra trees that exceed “area limit” parameter after sorting the trees according to their fitness value will be dropped
            self.forest = sorted(self.forest, key=itemgetter(self.dimension+1), reverse=True) # sort the forest according to the fitness from high to low
            self.forest = self.forest[:self.area_limit]  # 0~area_limit-1 : total area_limit


    def _global_seeding(self):
        selected_tree_num = int(self.transfer_rate * len(self.candidate_population))  # 有多少颗树进行 global seeding

        if selected_tree_num != 0:
            selected_trees_index = random.sample(range(len(self.candidate_population)), selected_tree_num)

            for index in selected_trees_index:
                temp_tree = self.candidate_population[index][:]
                selected_variables_index = random.sample(range(1, self.dimension+1), self.GSC)
                for i in selected_variables_index:
                    temp_tree[i] = 1 - temp_tree[i]  # The value of each selected variable will be negated (changing from 0 to 1 or vice versa)
                fitness = self._fitness_calculator(temp_tree[1: self.dimension+1])
                temp_tree[self.dimension+1] = fitness
                self.forest.append(temp_tree)


    def _update_best_tree(self):
        self.forest = sorted(self.forest, key=itemgetter(self.dimension+1), reverse=True) # sort the forest according to the fitness from high to low
        if (self.forest[0][self.dimension+1] > self.best_tree[self.dimension+1]) or \
           ((self.forest[0][self.dimension+1] == self.best_tree[self.dimension+1]) and \
            (Evaluation.get_dimension_reduction_rate(self.forest[0]) > Evaluation.get_dimension_reduction_rate(self.best_tree))):
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


    def _fitness_calculator(self, feature_subset):
        '''
        Returns
        ------------------------
        fitness: float
            classification accuracy
        '''

        fitness = Evaluation.k_nearest_neighbor(1, feature_subset, self.dataset, ValidationMethod.SEVEN_THREE)
#        fitness = Evaluation.svm_rbf(feature_subset, self.dataset, ValidationMethod.SEVEN_THREE)

##        print("-------feature_subset---------")
##        print(feature_subset)
##        print("-------fitness-------")
##        print(fitness)

        return fitness


    def evolution(self):
        '''
        Returns
        -----------------------
        best_tree: list
            best tree
        '''
        self._initialize_forest()

        for i in range(self.max_iterations):
            self._local_seeding()
            self._population_limiting()
            self._global_seeding()
            self._update_best_tree()

        return self.best_tree


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

        clf = svm.SVC(gamma='scale', kernel='rbf')

        return Evaluation._get_accuracy(clf, X, y, method)


    @staticmethod
    def cff_cart(feature_subset, dataset, method):
        X, y = Evaluation._getXy(feature_subset, dataset)

        clf = tree.DecisionTreeRegressor()

        return Evaluation._get_accuracy(clf, X, y, method)


    @staticmethod
    def _getXy(feature_subset, dataset):
        columns = dataset.shape[1]
        cols_to_use = [i for i in range(len(feature_subset)) if feature_subset[i] == 1]

        X = dataset[:, cols_to_use]
        y = dataset[:, columns-1]

        return [X, y]


    @staticmethod
    def _get_accuracy(clf, X, y, method):
        '''
        Parameters
        -----------------------
        clf:

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
        if X.size == 0:
            # if X = []. Caused by cols_to use = [], which means feature_subset is all zero, return 0 accuracy
            return accuracy

        accuracy = 0.0

        if method == ValidationMethod.SEVEN_THREE:
            #分割数据
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            clf.fit(X_train, y_train)  # 训练模型
            accuracy = clf.score(X_test, y_test)

        elif method == ValidationMethod.TEN_FOLD:
            accuracy = cross_val_accuracy(clf, X, y, cv=10, scoring='accuracy').mean()

        elif method == ValidationMethod.TWO_FOLD:
            accuracy = cross_val_accuracy(clf, X, y, cv=2, scoring='accuracy').mean()

        return accuracy


    @staticmethod
    def get_dimension_reduction_rate(feature_subset):
        cols_to_use = [i for i in range(len(feature_subset)) if feature_subset[i] == 1]
        selected_feature_num = len(cols_to_use)
        all_feature_num = len(feature_subset)
        return 1-(selected_feature_num/all_feature_num)


if __name__ == '__main__':
    start_time = time.time()
    file_path = ".\dataset\low\ionosphere.csv"
    forest = Forest(1, file_path, 100, 50, 15, 0.05)
    print(forest.evolution())
    print("--- %s seconds ---" % (time.time() - start_time))
