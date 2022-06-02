
import numpy as np
from math import log
import operator
import pickle
from tree_plot import createPlot
from sklearn.tree import DecisionTreeClassifier
from dataset import createDataSet, createDataSet2


class DecisionTree():
    def __init__(self, criterion='gini', total_feats=None, id2feat_mapping=None, if_plot=True):
        """
        :param criterion: str, optional['gini', 'entropy']
            Criterion to split data.
        :param total_feats: List[str]
            Feature name list.
        :param id2feat_mapping: dict[...]
            Feature id to feature name mapping dict.
        :param if_plot: if plot decision tree.
        """
        self.criteriion = criterion
        self.feats_list = total_feats
        self.id2feat_mapping = id2feat_mapping
        self.label2id_mapping = None
        if id2feat_mapping is not None:
            self.label2id_mapping = {id2feat_mapping['label'][l_id]: l_id for l_id in id2feat_mapping['label']}
        self.if_plot = if_plot
        self.tree = None
        self.feat_type = []
        self.epsilon = 1e-7
        self.feats_pairs = None

    def fit(self, x, y):
        """
        Train decision tree model.
        :param x: np.ndarray, [train_num, feat_dim]
        :param y: np.ndarray, [train_num]
        """
        assert type(x).__name__ == 'ndarray', 'the type of input x must be np.ndarray !'
        assert len(x.shape) == 2, 'the shape of input x must be 2D !'
        assert type(y).__name__ == 'ndarray', 'the type of input y must be np.ndarray !'

        if not self.feats_list:
            self.feats_list = [('feat_%d' % i) for i in range(x.shape[1])]
        if not self.id2feat_mapping:
            # feature mapping
            id2feat_mapping = {}
            for j in range(x.shape[1]):
                id2featvalue = {}
                feat_name = self.feats_list[j]
                unique_feat_value = set(list(x[:, j]))
                for v in unique_feat_value:
                    id2featvalue[v] = 'value_{}'.format(v)
                id2feat_mapping[feat_name] = id2featvalue
            # label mapping
            label_mapping = {}
            unique_label = set(list(y.reshape(-1)))
            for label in unique_label:
                label_mapping[label] = 'label_{}'.format(label)
            id2feat_mapping['label'] = label_mapping
            self.id2feat_mapping = id2feat_mapping
            self.label2id_mapping = {id2feat_mapping['label'][l_id]: l_id for l_id in id2feat_mapping['label']}

        # check feature type
        for col in range(x.shape[1]):
            self.check_data_type(x[:, col])
        self.feats_pairs = [(feat, self.feat_type[i]) for i, feat in enumerate(self.feats_list)]
        feats_pairs = self.feats_pairs[:]  # copy feat list
        y = y.reshape(-1, 1)
        data = np.concatenate([x, y], axis=1)

        self.tree = self.createTree(data, feats_pairs,
                                    criterion=self.criteriion,
                                    id2feat_mapping=self.id2feat_mapping)
        if self.if_plot:
            createPlot(self.tree)

    def predict(self, test_x):
        """
        :param test_x: np.ndarray, [test_num, feat_dim]
        :return: predicted results, [test_num]
        """
        assert type(test_x).__name__ == 'ndarray', 'test_x must be np.ndarray !'
        assert 1 <= len(test_x.shape) <= 2, 'test_x must be 1D or 2D!'
        if len(test_x.shape) == 1:
            test_x = test_x.reshape(-1, len(self.feats_list))
        pred = []
        for i in range(test_x.shape[0]):
            test_vec = test_x[i]
            pred_label = self.predict_(test_vec, self.tree)
            pred.append(self.label2id_mapping[pred_label])
        return np.array(pred)

    def score(self, test_x, target):
        """
        :param test_x: np.ndarray, [test_num, feat_dim]
        :param target: np.ndarray, [test_num]
        :return: accuracy in test data, float.
        """
        assert type(test_x).__name__ == 'ndarray', 'The type of input x must be np.ndarray !'
        assert len(test_x.shape) == 2, 'The shape of input x must be 2D !'
        pred_y = self.predict(test_x)
        return self.calc_accuracy(pred_y, target)

    def predict_(self, test_vec, tree=None):
        """
        :param test_vec:  np.ndarray, [feat_dim]
        """
        feat_name = next(iter(tree))  # tree root which stores a feature
        feat_values = tree[feat_name]  # subgraph representing different values of the same feature
        featIndex = self.feats_list.index(feat_name)
        classLabel = None
        feat_index = self.feats_list.index(feat_name)
        if self.feat_type[feat_index] == 'int':
            cur_value = test_vec[featIndex]
            for key in feat_values.keys():
                if self.id2feat_mapping[feat_name][cur_value] == key:
                    if type(feat_values[key]).__name__ == 'dict':  # if not a leaf node
                        classLabel = self.predict_(test_vec, feat_values[key])
                    else:
                        classLabel = feat_values[key]
            if not classLabel:
                raise Exception('Can find value:{} of feature:{} '.format(cur_value, feat_name))
        elif self.feat_type[feat_index] == 'float':
            cur_value = test_vec[featIndex]
            for divide_str in feat_values.keys():
                oper_str = divide_str[0]
                if oper_str == '<' and cur_value > float(divide_str[-5:]):
                    continue
                if oper_str == '>' and cur_value <= float(divide_str[-5:]):
                    continue
                if type(feat_values[divide_str]).__name__ == 'dict':
                    classLabel = self.predict_(test_vec, feat_values[divide_str])
                else:
                    classLabel = feat_values[divide_str]

        return classLabel

    def createTree(self, dataSet, candFeats, fatherClass=None, criterion='gini', id2feat_mapping=None):
        """
        Create decision tree.
        :param dataSet: [data_num, left_feat_dim + label]
            Data in current tree node.
        :param candFeats: list[(feat_name, feat_type)]
            Candidate features, also means features which have not been used.
        :param fatherClass: [father_data_num]
            Class list in father node.
        :param criterion: str
            Criterion to splitting data.
        :param id2feat_mapping: dict[dict]
            Feature id to name mapping dict.
        :return: dict[dict[...]]
            Decision Tree stared by dict.
        """
        classList = dataSet[:, -1]  # label, [data_num]

        # return case (1)
        # if only on class left
        if (classList == classList[0]).sum() == len(classList):
            return id2feat_mapping['label'][classList[0]]
        # return case (2)
        # if feature set is empty or all data are the identical values in the same feature
        if len(candFeats) == 0 or self.isSameData(dataSet):
            major_target = self.majorityCnt(classList)
            return id2feat_mapping['label'][major_target]  # return class that appears most times in the data
        # return case (3)
        # if the sample set of the current node is empty
        if dataSet.size == 0:
            try:
                father_major_target = self.majorityCnt(fatherClass)
                return id2feat_mapping['label'][
                    father_major_target]  # use class which appears most times in father node
            except:
                raise Exception('Class list in father root can not be None! Except at the root node.')
        # choose the best splitting feature
        bestFeatType, bestFeatId, bestValue = self.chooseBestFeature(dataSet, candFeats, criterion)
        bestFeatLabel = candFeats[bestFeatId][0]
        myTree = {bestFeatLabel: {}}  # current tree
        if bestFeatType == 'int':
            del (candFeats[bestFeatId])  # delete best feature from candidate features
            featValues = dataSet[:, bestFeatId]  # [data_num]
            uniqueVals = set(featValues)  # de-duplication
            for value in uniqueVals:
                subCandFeats = candFeats[:]  # copy candidate features
                # recursion
                feat_value_name = id2feat_mapping[bestFeatLabel][value]
                subData = self.splitDataSetDiscrete(dataSet, bestFeatId, value)
                myTree[bestFeatLabel][feat_value_name] = self.createTree(subData, subCandFeats,
                                                                         fatherClass=classList,
                                                                         criterion=criterion,
                                                                         id2feat_mapping=id2feat_mapping)
        elif bestFeatType == 'float':
            divideEdges = ['<=', '>']
            for idx, subData in enumerate(self.splitDataSetContinuous(dataSet, bestFeatId, bestValue)):
                subCandFeats = candFeats[:]  # copy
                feat_value_name = '%s%.3f' % (divideEdges[idx], bestValue)
                myTree[bestFeatLabel][feat_value_name] = self.createTree(subData, subCandFeats,
                                                                         fatherClass=classList,
                                                                         criterion=criterion,
                                                                         id2feat_mapping=id2feat_mapping)
        else:
            raise Exception('Valid feature type !')

        return myTree

    def calcEntropy(self, dataSet):
        """
        calculate entropy in current data set by class.
        :param dataSet: [data_num, feat_dim]
        :return: entropy, float
        """
        numEntires = len(dataSet)  # data size
        labelCounts = {}
        for featVec in dataSet:
            currentLabel = featVec[-1]  # current sample's label
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / numEntires
            shannonEnt -= prob * log(prob, 2)
        return shannonEnt

    def calcGini(self, dataSet):
        """
        Calculate Gini coefficient.
        """
        numEntires = len(dataSet)  # data size
        labelCounts = {}
        for featVec in dataSet:
            currentLabel = featVec[-1]  # current sample's label
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        gini = 1
        for key in labelCounts:
            prob = float(labelCounts[key]) / numEntires
            gini -= prob ** 2
        return gini

    def splitDataSetDiscrete(self, dataSet, axis, value):
        """
        Split data set by specified value of chose feature.
        :param dataSet: data set, [data_num, feat_dim]
        :param axis: splitting feature id, int.
        :param value: splitting feature value, str.
        :return: The data satisfying the feature V which takes the value of v, [split_data_num, feat_dim - 1]
        """
        retDataSet = []
        for featVec in dataSet:
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec = np.concatenate([reducedFeatVec, featVec[axis + 1:]], axis=0)
                retDataSet.append(reducedFeatVec)
        retDataSet = np.array(retDataSet)
        return retDataSet

    def splitDataSetContinuous(self, dataSet, axis, value):
        """
        Split data set by specified value of chose feature.
        :param dataSet: data set, [data_num, feat_dim]
        :param axis: splitting feature id, int.
        :param value: splitting feature value, str.
        """
        splitDataSets = []
        # feature value less than specified value
        index_less = dataSet[:, axis] <= value
        splitDataSets.append(dataSet[index_less])
        # feature value great than specified value
        index_great = dataSet[:, axis] > value
        splitDataSets.append(dataSet[index_great])
        return splitDataSets

    def chooseBestFeatureByEntropy(self, dataSet, candFeats):
        """
        Choose best feature by maximum information gain.
        Gain(D,a) = Ent(D) - sum_{v=1}^V |D^v|/|D|*Ent(D^v)

        :param dataSet: split data in current tree node, [data_num, left_feat_dim]
        :return: best feature id in current split data.
        """
        numFeatures = len(dataSet[0]) - 1  # feature size V
        baseEntropy = self.calcEntropy(dataSet)  # Ent(D)
        bestInfoGain = 0.0
        bestFeatureId = -1  # id of beat feature
        bestDivideValue = None
        for i in range(numFeatures):
            feat_type = candFeats[i][1]
            featList = [example[i] for example in dataSet]  # each feature value in data, [data_num]
            uniqueVals = set(featList)  # de-duplication
            if feat_type == 'int':
                subEntropy = 0.0  # entropy in split data
                for value in uniqueVals:
                    subDataSet = self.splitDataSetDiscrete(dataSet, i, value)  # split data
                    prob = len(subDataSet) / float(len(dataSet))
                    subEntropy += prob * self.calcEntropy(subDataSet)
                infoGain = baseEntropy - subEntropy
                if infoGain > bestInfoGain:
                    bestInfoGain = infoGain
                    bestFeatureId = i
            elif feat_type == 'float':
                sortedUniquevals = sorted(list(uniqueVals))
                for idx in range(len(sortedUniquevals) - 1):
                    subEntropy = 0.0  # entropy in split data
                    divide_value = (sortedUniquevals[idx] + sortedUniquevals[idx + 1]) / 2
                    for subData in self.splitDataSetContinuous(dataSet, i, divide_value):
                        prob = len(subData) / float(len(dataSet))
                        subEntropy += prob * self.calcEntropy(subData)
                    infoGain = baseEntropy - subEntropy
                    if infoGain > bestInfoGain:
                        bestInfoGain = infoGain
                        bestFeatureId = i
                        bestDivideValue = divide_value
            else:
                raise Exception("Invalid feature type !")
        return candFeats[bestFeatureId][1], bestFeatureId, bestDivideValue

    def chooseBestFeatureByGini(self, dataSet, candFeats):
        """
        Choose best feature by minimum gini coefficient.
        Gini_index(D,a) = sum_{v=1}^V |D^v|/|D|*Gini(D^v)
        :param candFeats: list[(feat_name, feat_type)]
            Candidate feature list.
        :param dataSet: data in current tree node, [data_num, left_feat_dim]
        :return: best splitting feature id in current data.
        """
        numFeatures = len(dataSet[0]) - 1  # feature size
        minGini = float('inf')
        bestFeatureId = None  # id of beat feature
        bestDivideValue = None
        for i in range(numFeatures):
            feat_type = candFeats[i][1]
            featList = dataSet[:, i]  # each feature value in data, [data_num]
            uniqueVals = set(featList)  # de-duplication
            if feat_type == 'int':
                gini = 0.0
                for value in uniqueVals:
                    subDataSet = self.splitDataSetDiscrete(dataSet, i, value)  # split data
                    prob = len(subDataSet) / float(len(dataSet))
                    gini += prob * self.calcGini(subDataSet)
                if gini < minGini:
                    minGini = gini
                    bestFeatureId = i
            elif feat_type == 'float':
                sortedUniquevals = sorted(list(uniqueVals))
                for idx in range(len(uniqueVals) - 1):
                    gini = 0.0
                    divide_value = (sortedUniquevals[idx] + sortedUniquevals[idx + 1]) / 2
                    for subData in self.splitDataSetContinuous(dataSet, i, divide_value):
                        prob = len(subData) / float(len(dataSet))
                        gini += prob * self.calcGini(subData)
                    if gini < minGini:
                        minGini = gini
                        bestFeatureId = i
                        bestDivideValue = divide_value
            else:
                raise Exception("Invalid feature type !")
        return candFeats[bestFeatureId][1], bestFeatureId, bestDivideValue

    def chooseBestFeature(self, dataSet, candFeats, criterion='gini'):
        if criterion == 'entropy':
            return self.chooseBestFeatureByEntropy(dataSet, candFeats)
        if criterion == 'gini':
            return self.chooseBestFeatureByGini(dataSet, candFeats)

    def majorityCnt(self, classList):
        """
        Return class which appears most times in current data.
        :param classList: [data_num, feat_dim]
        :return:
        """
        assert classList is not None, 'classList can not be None !'
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

        return sortedClassCount[0][0]

    def isSameData(self, dataSet):
        data_num = len(dataSet)
        for i in range(1, data_num):
            if np.any(dataSet[i - 1, :-1] != dataSet[i, :-1]):
                return False
        return True

    def check_data_type(self, feat_values):
        """
        :param feat_values: [data_num]
        """
        feat_values_int = feat_values.astype('int')
        dis = abs(feat_values_int.astype('float64') - feat_values).sum()
        if dis < self.epsilon:
            self.feat_type.append('int')
            return 'int'
        self.feat_type.append('float')
        return 'float'

    def calc_accuracy(self, pred, target):
        """
        :param pred: np.ndarray, [test_size]
            Predicted labels.
        :param target: np.ndarray, [test_size]
            Ground truth targets.
        :return: accuracy, float
        """
        assert pred.shape == target.shape, 'The shape of pred must equals to target!'

        return (pred == target).sum() / float(len(target))


if __name__ == '__main__':
    x, y, total_feats, id2feat_mapping = createDataSet2()
    test_id = -4
    x_train, y_train = x, y
    x_test, y_test = x[test_id, :], y[test_id].reshape(1)
    print('total train size:', len(x_train))

    model = DecisionTree(criterion='entropy', total_feats=total_feats, id2feat_mapping=id2feat_mapping)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    test_label = [id2feat_mapping['label'][ele] for ele in y_test]
    pred_label = [id2feat_mapping['label'][ele] for ele in pred]
    print('true label:', y_test, test_label)
    print('pred label:', pred, pred_label)

