
import numpy as np


def createDataSet():
    """
    Watermelon data set only contains discrete features.
    Including 17 data instances with 6 features.

    Returns
    -------
    x: np.ndarray, [data_num, feature_num]
    y: np.ndarray, [data_num]
    labels: List[str], len = feature_num
        List contains feature names.
    id2feat_mapping: dict
        Id to feature name mapping dict.
    """
    dataSet = [
        # 1
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        # 2
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        # 3
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        # 4
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        # 5
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        # 6
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
        # 7
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
        # 8
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],

        # ----------------------------------------------------
        # 9
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
        # 10
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
        # 11
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
        # 12
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
        # 13
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
        # 14
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
        # 15
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
        # 16
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
        # 17
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
    ]

    # 特征值列表
    labels = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感']
    id2feat_mapping = {'色泽': {0: '青绿', 1: '乌黑', 2: '浅白'},
                       '根蒂': {0: '蜷缩', 1: '稍蜷', 2: '硬挺'},
                       '敲击': {0: '浊响', 1: '沉闷', 2: '清脆'},
                       '纹理': {0: '清晰', 1: '稍糊', 2: '模糊'},
                       '脐部': {0: '凹陷', 1: '稍凹', 2: '平坦'},
                       '触感': {0: '硬滑', 1: '软粘'},
                       'label': {0: '坏瓜', 1: '好瓜'}}
    feat2id_mapping = {}

    for feat_name in id2feat_mapping.keys():
        sub_dict = {}
        for feat_id_value in id2feat_mapping[feat_name].keys():
            sub_dict[id2feat_mapping[feat_name][feat_id_value]] = feat_id_value
        feat2id_mapping[feat_name] = sub_dict

    for i in range(len(dataSet)):
        for j in range(len(dataSet[0])):
            if j == len(dataSet[0]) - 1:
                dataSet[i][j] = 1 if dataSet[i][j] == '好瓜' else 0
                continue
            feat_name = labels[j]
            feat_value = dataSet[i][j]
            dataSet[i][j] = feat2id_mapping[feat_name][feat_value]

    dataSet = np.array(dataSet)
    x = dataSet[:, :-1]
    y = dataSet[:, -1]

    return x, y, labels, id2feat_mapping


def createDataSet2():
    """
    Watermelon data set contains discrete and continuos features.
    Including 17 data instances with 8 features.

    Returns
    -------
    x: np.ndarray, [data_num, feature_num]
    y: np.ndarray, [data_num]
    labels: List[str], length = feature_num
        List contains feature names.
    id2feat_mapping: dict
        Id to feature name mapping dict.
    """
    dataSet = [
        # 1
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '好瓜'],
        # 2
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '好瓜'],
        # 3
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '好瓜'],
        # 4
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '好瓜'],
        # 5
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '好瓜'],
        # 6
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '好瓜'],
        # 7
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '好瓜'],
        # 8
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '好瓜'],

        # ----------------------------------------------------
        # 9
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '坏瓜'],
        # 10
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '坏瓜'],
        # 11
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '坏瓜'],
        # 12
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '坏瓜'],
        # 13
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '坏瓜'],
        # 14
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '坏瓜'],
        # 15
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '坏瓜'],
        # 16
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '坏瓜'],
        # 17
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '坏瓜'],
    ]

    # 特征值列表
    labels = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率']

    id2feat_mapping = {'色泽': {0: '青绿', 1: '乌黑', 2: '浅白'},
                       '根蒂': {0: '蜷缩', 1: '稍蜷', 2: '硬挺'},
                       '敲击': {0: '浊响', 1: '沉闷', 2: '清脆'},
                       '纹理': {0: '清晰', 1: '稍糊', 2: '模糊'},
                       '脐部': {0: '凹陷', 1: '稍凹', 2: '平坦'},
                       '触感': {0: '硬滑', 1: '软粘'},
                       'label': {0: '坏瓜', 1: '好瓜'}}
    feat2id_mapping = {}

    for feat_name in id2feat_mapping.keys():
        sub_dict = {}
        for feat_id_value in id2feat_mapping[feat_name].keys():
            sub_dict[id2feat_mapping[feat_name][feat_id_value]] = feat_id_value
        feat2id_mapping[feat_name] = sub_dict

    for i in range(len(dataSet)):
        for j in range(len(dataSet[0])):
            if j == len(dataSet[0]) - 2 or j == len(dataSet[0]) - 3:
                continue
            if j == len(dataSet[0]) - 1:
                dataSet[i][j] = 1 if dataSet[i][j] == '好瓜' else 0
                continue
            feat_name = labels[j]
            feat_value = dataSet[i][j]
            dataSet[i][j] = feat2id_mapping[feat_name][feat_value]

    dataSet = np.array(dataSet)
    x = dataSet[:, :-1]
    y = dataSet[:, -1].astype('int')

    return x, y, labels, id2feat_mapping


if __name__ == '__main__':
    x, y, labels, id2feat_mapping = createDataSet2()
    print(x.shape)
    print(y.shape)



