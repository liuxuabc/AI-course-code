import numpy as np
import math
import random
import matplotlib.pyplot as plt
import networkx as nx

#numbers为需要经过的其他城市的数量
def initialization(numbers):
    path_random = np.random.choice(list(range(1, numbers + 1)), replace=False, size=numbers)
    path_random = path_random.tolist()
    return path_random

def generate_new(path):
    numbers = len(path)
    #随机生成两个不重复的点
    positions = np.random.choice(list(range(numbers)), replace=False, size=2)
    lower_position = min(positions[0], positions[1])
    upper_position = max(positions[0], positions[1])
    #将数列中段逆转
    mid_reversed = path[lower_position:upper_position + 1]
    mid_reversed.reverse()
    #拼接生成新的数列
    new_path = path[:lower_position]
    new_path.extend(mid_reversed)
    new_path.extend(path[upper_position + 1:])
    return new_path

#length_mat为各个节点之间的最短距离矩阵
def e(path, length_mat):
    numbers = len(path) - 1
    length = length_mat[0][path[0]] + length_mat[path[-1]][0]
    for i in range(numbers):
        length += length_mat[path[i]][path[i + 1]]
    return length

def metropolis(e, new_e, t):
    if new_e <= e:
        return True
    else:
        p = math.exp((e - new_e) / t)
        return True if random.random() < p else False

def search(all_path, length_mat):
    best_e = 0xffff
    best_path = all_path[0]
    for path in all_path:
        ex = e(path, length_mat)
        if ex < best_e:
            best_e = ex
            best_path = path
    return best_path

#t0为初始温度，t_final为终止温度，alpha为冷却系数，inner_iter为内层迭代次数
#city_numbers为所需要经过的其他城市，length_mat为各个节点之间的最短距离矩阵
def sa(t0, t_final, alpha, inner_iter, city_numbers, length_mat):
    all_path = []
    all_ex = []
    init = initialization(city_numbers)
    all_path.append(init)
    all_ex.append(e(init, length_mat))
    t = t0
    while t > t_final:
        path = search(all_path, length_mat)
        ex = e(path, length_mat)
        for i in range(inner_iter):
            new_path = generate_new(path)
            new_ex = e(new_path, length_mat)
            if metropolis(ex, new_ex, t):
                path = new_path
                ex = new_ex
        all_path.append(path)
        all_ex.append(ex)
        t = alpha * t
    return all_path, all_ex

graph = nx.Graph()
graph.add_nodes_from(range(0, 5))
edges = [(0, 1, 1165), (0, 2, 1462),
         (0, 3, 3179), (0, 4, 1967), (1, 2, 1511),
         (1, 3, 1942), (1, 4, 2129), (2, 3, 2677), (2, 4, 1181),
         (3, 4, 2216)]
graph.add_weighted_edges_from(edges)
shortest_length = dict(nx.shortest_path_length(graph, weight='weight'))
length_mat = []
for i in range(0, 5):
    i_j = []
    for j in range(0, 5):
        i_j.append(shortest_length[i][j])
    length_mat.append(i_j)

all_path, all_ex = sa(5000, pow(10, -1), 0.98, 200, 4, length_mat)
print(search(all_path, length_mat))
print('距离总和是',round(e(search(all_path, length_mat), length_mat)))
iteration = len(all_path)
all_path = np.array(all_path)
all_ex = np.array(all_ex)
plt.xlabel("Iteration")
plt.ylabel("Length")
plt.plot(range(iteration), all_ex)
plt.show()
