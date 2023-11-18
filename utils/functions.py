# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2021-04-22 11:36


def logistic_function(x, sigma):
    for i in range(len(x)):
        x[i] = sigma * x[i] * (1 - x[i])
    return x


class Permutation(object):
    def __init__(self):
        self.results = []

    def fulfill_all_permutation(self, array):
        self.__sub_perm(array, 0)

    def __sub_perm(self, array, start):
        if start == len(array):
            self.results.append([item for item in array])
        for i in range(start, len(array)):
            array[start], array[i] = array[i], array[start]
            self.__sub_perm(array, start + 1)
            array[start], array[i] = array[i], array[start]

    def get_all_permutation(self, array):
        if len(self.results) < 1:
            self.__sub_perm(array, 0)
        return self.results


def edit_distance(series1, series2) -> int:
    row, col = len(series1), len(series2)
    dp = [[0 for _ in range(col + 1)] for _ in range(row + 1)]
    for i in range(1, row + 1):
        dp[i][0] = i
    for i in range(1, col + 1):
        dp[0][i] = i
    for i in range(1, row + 1):
        for j in range(1, col + 1):
            if series1[i - 1] == series2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
    return dp[row][col]


if __name__ == '__main__':
    # po = Permutation()
    # po.fulfill_all_permutation([1, 2, 3, 4])
    # for res in po.results:
    #     print(res)
    print(edit_distance([1, 2, 3, 4], [4, 3, 2, 1]))
