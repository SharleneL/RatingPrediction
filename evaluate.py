__author__ = 'luoshalin'


def cal_accuracy(list1, list2):
    cnt = 0
    length = 0
    for i, j in zip(list1, list2):
        length += 1
        if i == j:
            cnt += 1

    return float(cnt) / length