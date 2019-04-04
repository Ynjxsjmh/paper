# import .evaluation
from enum import Enum, unique
from sklearn import svm
from sklearn import tree as Tree
# K最近邻(kNN，k-NearestNeighbor)分类算法
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score   # K折交叉验证模块
from sklearn.model_selection import train_test_split  # 分割数据模块

# ------------- Evaluation. py --------------------------------
@unique               # 该装饰器可以帮助我们检查保证没有重复值
class ValidationMethod(Enum):
    TEN_FOLD = 1      # 10-fold
    TWO_FOLD = 2      # 2-fold
    SEVEN_THREE = 3   # 70%-30%


@unique
class EvaluationFunction(Enum):
    ONE_NN = 1      # 1-NN
    THREE_NN = 2
    FIVE_NN = 3
    SVM_RBF = 4
    C45_CART = 5


class Evaluation(object):
    @staticmethod
    def _k_nearest_neighbor(k, X, y, method):
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
        knn = KNeighborsClassifier(n_neighbors=k)  # 建立模型

        return Evaluation._get_accuracy(knn, X, y, method)

    @staticmethod
    def _svm_rbf(X, y, method):
        clf = svm.SVC(gamma='scale', kernel='rbf')

        return Evaluation._get_accuracy(clf, X, y, method)

    @staticmethod
    def _cff_cart(X, y, method):
        clf = Tree.DecisionTreeClassifier()

        return Evaluation._get_accuracy(clf, X, y, method)

    @staticmethod
    def _getXy(feature_subset, dataset):
        columns = dataset.shape[1]
        cols_to_use = [
            i for i in range(len(feature_subset))
            if feature_subset[i] == 1]

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
            accuracy = cross_val_score(
                clf, X, y, cv=10, scoring='accuracy').mean()

        elif method == ValidationMethod.TWO_FOLD:
            accuracy = cross_val_score(
                clf, X, y, cv=2, scoring='accuracy').mean()

        return accuracy

    @staticmethod
    def get_dimension_reduction_rate(feature_subset):
        cols_to_use = [
            i for i in range(len(feature_subset))
            if feature_subset[i] == 1]
        selected_feature_num = len(cols_to_use)
        all_feature_num = len(feature_subset)
        return 1 - (selected_feature_num / all_feature_num)

    @staticmethod
    def get_accuracy(eval_function, feature_subset, dataset, method):
        X, y = Evaluation._getXy(feature_subset, dataset)

        accuracy = 0.0
        if eval_function == EvaluationFunction.ONE_NN:
            accuracy = Evaluation._k_nearest_neighbor(1, X, y, method)
        elif eval_function == EvaluationFunction.THREE_NN:
            accuracy = Evaluation._k_nearest_neighbor(3, X, y, method)
        elif eval_function == EvaluationFunction.FIVE_NN:
            accuracy = Evaluation._k_nearest_neighbor(5, X, y, method)
        elif eval_function == EvaluationFunction.SVM_RBF:
            accuracy = Evaluation._svm_rbf(X, y, method)
        elif eval_function == EvaluationFunction.C45_CART:
            accuracy = Evaluation._cff_cart(X, y, method)

        return accuracy

# ------------- Evaluation. py --------------------------------
