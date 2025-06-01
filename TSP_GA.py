#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2025/5/31 10:40
# @Author: ZhaoKe
# @File : TSP_GA.py
# @Software: PyCharm
# import copy
import copy
import random
import numpy as np
import matplotlib.pyplot as plt


class Solution(object):
    def __init__(self, vector, value):
        self.vector = vector
        self.value = value

    def __len__(self):
        return len(self.vector)

    def __str__(self):
        return '[' + ','.join(self.vector) + '], ' + str(self.value)


class TSPGA(object):
    def __init__(self, city_data):
        self.population_num = 100
        self.max_iter = 200
        self.pc, self.pm = 0.9, 0.1
        self.ps = 10
        self.city_data = city_data
        self.solution_length = len(city_data)
        self.solution_list = []
        self.distance_M = None
        self.__calculate_distance()
        self.best_solution = None

    def __calculate_distance(self):
        self.distance_M = np.zeros((self.solution_length, self.solution_length))
        for i in range(self.solution_length):
            for j in range(i + 1, self.solution_length):
                self.distance_M[i, j] = np.sqrt((self.city_data[i][0] - self.city_data[j][0]) ** 2 + (
                        self.city_data[i][1] - self.city_data[j][1]) ** 2)
        self.distance_M = self.distance_M + self.distance_M.T

    def calculate_fitness(self, solution):
        fitness_v = 0
        for i in range(len(solution) - 1):
            fitness_v += self.distance_M[solution[i], solution[i + 1]]
        return fitness_v

    def __generate_solution(self):
        temp_s = list(range(self.solution_length))
        random.shuffle(temp_s)
        return temp_s

    def init_population(self):
        for _ in range(self.population_num):
            temp_vec = self.__generate_solution()
            temp_val = self.calculate_fitness(temp_vec)
            self.solution_list.append(Solution(temp_vec, temp_val))

    def mutation(self, solution):
        st = random.randint(0, self.solution_length - 1)
        en = random.randint(st, self.solution_length - 1)
        tmp = solution[st]
        solution[st] = solution[en]
        solution[en] = tmp
        return solution

    def crossover(self, S1):
        # Step 1: 截断 S2
        S2 = random.choice(self.solution_list).vector
        p = random.randint(int(0.2 * self.solution_length), int(0.8 * self.solution_length))
        tail = S2[p:]
        S2_part = S2[:p]

        # Step 2: 找出 tail 中元素在 S1 中的原始索引
        original_S1 = S1.copy()
        indices = [original_S1.index(x) for x in tail]

        # Step 3: 从 S1 中删除 tail 中的元素
        elements_to_remove = set(tail)
        S1_temp = [x for x in original_S1 if x not in elements_to_remove]

        # Step 4: 按照原始索引插入 tail 中的元素到 S2_part
        insert_list = sorted(zip(indices, tail), key=lambda x: x[0])
        for idx, x in insert_list:
            S2_part.insert(idx, x)

        # Step 5: 拼接 S1_temp 和 tail，形成新的 S1
        S1_new = S1_temp + tail
        S2_new = S2_part

        return S1_new, S2_new

    def main(self):
        print("init population")
        self.init_population()

        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter([c[0] for c in self.city_data], [c[1] for c in self.city_data], c='blue')
        for i, (x, y) in enumerate(cities):
            ax.text(x, y, str(i), fontsize=10)
        ax.set_title("TSP Path Evolution")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        plt.show()

        # print("")
        # plt.ion()
        for iter_idx in range(self.max_iter):

            self.solution_list.sort(key=lambda x: x.value, reverse=False)
            self.best_solution = self.solution_list[0]
            print(self.best_solution.value)

            if self.best_solution is None:
                self.best_solution = self.solution_list[0]
            elif self.best_solution.value > self.solution_list[0].value:
                self.best_solution = self.solution_list[0]
            print(self.best_solution.value)

            ax.clear()
            # 绘制城市点
            ax.scatter([c[0] for c in cities], [c[1] for c in cities], c='blue')
            for i, (x, y) in enumerate(cities):
                ax.text(x, y, str(i), fontsize=10)

            # 绘制路径线
            path_coords = [cities[i] for i in self.best_solution.vector]
            path_coords.append(path_coords[0])  # 回到起点
            xs, ys = zip(*path_coords)
            ax.plot(xs, ys, 'r--', lw=2)

            # 显示信息
            ax.set_title(f"Generation {iter_idx} | Best Distance: {self.best_solution.value:.2f}")
            ax.set_xlim(min([c[0] for c in cities]) - 1, max([c[0] for c in cities]) + 1)
            ax.set_ylim(min([c[1] for c in cities]) - 1, max([c[1] for c in cities]) + 1)
            ax.grid(True)

            plt.draw()
            plt.pause(0.1)  # 控制更新频率

            prob_list = np.array([x.value for x in self.solution_list])
            p_sum = prob_list.sum()
            p_item = prob_list / p_sum
            solutions_prob = [p_item[0]]
            for i in range(1, len(p_item)):
                solutions_prob.append(solutions_prob[-1] + p_item[i])
            solutions_prob = solutions_prob[::-1]

            for i in range(self.population_num):
                if random.random() > solutions_prob[i]:
                    if random.random() < self.pc:
                        c_s1, c_s2 = self.crossover(self.solution_list[i].vector)
                        v1, v2 = self.calculate_fitness(c_s1), self.calculate_fitness(c_s2)
                        if v1 < v2:
                            self.solution_list[i] = Solution(c_s1, v1)
                        else:
                            self.solution_list[i] = Solution(c_s2, v2)
                    if random.random() < self.pm:
                        m_s = self.mutation(self.solution_list[i].vector)
                        m_s_v = self.calculate_fitness(m_s)
                        self.solution_list[i] = Solution(m_s, m_s_v)
            self.solution_list.sort(key=lambda x: x.value, reverse=False)
            # # self.select()
            # break
        self.solution_list.sort(key=lambda x: x.value, reverse=False)
        self.best_solution = self.solution_list[0]
        print(self.best_solution.value)

        if self.best_solution is None:
            self.best_solution = self.solution_list[0]
        elif self.best_solution.value > self.solution_list[0].value:
            self.best_solution = self.solution_list[0]
        print(self.best_solution.value)
        ax.clear()
        # 绘制城市点
        ax.scatter([c[0] for c in cities], [c[1] for c in cities], c='blue')
        for i, (x, y) in enumerate(cities):
            ax.text(x, y, str(i), fontsize=10)

        # 绘制路径线
        path_coords = [cities[i] for i in self.best_solution.vector]
        path_coords.append(path_coords[0])  # 回到起点
        xs, ys = zip(*path_coords)
        ax.plot(xs, ys, 'r--', lw=2)

        # 显示信息
        ax.set_title(f"Generation {self.max_iter} | Best Distance: {self.best_solution.value:.2f}")
        ax.set_xlim(min([c[0] for c in cities]) - 1, max([c[0] for c in cities]) + 1)
        ax.set_ylim(min([c[1] for c in cities]) - 1, max([c[1] for c in cities]) + 1)
        ax.grid(True)

        plt.draw()
        plt.show()
        plt.savefig("./TSP_10cities.png")


if __name__ == '__main__':
    cities = [
        (2066, 2333),
        (935, 1304),
        (1270, 200),
        (1389, 700),
        (984, 2810),
        (2253, 478),
        (949, 3025),
        (87, 2483),
        (3094, 1883),
        (2706, 3130),
    ]
    tspga_obj = TSPGA(cities)
    tspga_obj.main()

    # print(tspga_obj.distance_M)
    # tspga_obj.init_population()

    # s1 = [1, 3, 2, 4, 7, 6, 5, 8, 9, 0]
    # v = tspga_obj.calculate_fitness(s1)
    # s1 = Solution(s1, v)
    # s2 = [0, 1, 3, 6, 2, 4, 7, 5, 8, 9]
    # v = tspga_obj.calculate_fitness(s2)
    # s2 = Solution(s2, v)
    # tspga_obj.solution_list.append(s2)
    # ns1, ns2 = tspga_obj.crossover(s1.vector)
    # print(ns1)
    # print(ns2)
