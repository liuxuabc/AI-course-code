
import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import random



# 准备好距离矩阵
city_num = 5
city_dist_mat = np.zeros([city_num, city_num])
city_dist_mat[0][1] = city_dist_mat[1][0] = 1165
city_dist_mat[0][2] = city_dist_mat[2][0] = 1462
city_dist_mat[0][3] = city_dist_mat[3][0] = 3179
city_dist_mat[0][4] = city_dist_mat[4][0] = 1967
city_dist_mat[1][2] = city_dist_mat[2][1] = 1511
city_dist_mat[1][3] = city_dist_mat[3][1] = 1942
city_dist_mat[1][4] = city_dist_mat[4][1] = 2129
city_dist_mat[2][3] = city_dist_mat[3][2] = 2677
city_dist_mat[2][4] = city_dist_mat[4][2] = 1181
city_dist_mat[3][4] = city_dist_mat[4][3] = 2216
# 标号说明
# list_city = ['0_北京', '1_西安', '2_上海', '3_昆明', '4_广州']
# 1.定义个体类，包括基因（城市路线）和适应度

num_person_idx = 0
num_person = 0
dis_list = []


class Individual:
    def __init__(self, genes=None):
        global num_person
        global dis_list
        global num_person_idx
        num_person_idx += 1
        if num_person_idx % 20 == 0:
            num_person += 1
        self.genes = genes
        if self.genes == None:
            genes = [0] * 5
            temp = [0] * 4
            temp = [i for i in range(1, city_num)]
            random.shuffle(temp)
            genes[1:] = temp
            genes[0] = 0
            self.genes = genes
            # print("init_self.genes = ",self.genes)
            self.fitness = self.evaluate_fitness()
            # self.fitness = fitness
            # dis_list.append(-1.0)
        else:
            self.fitness = float(self.evaluate_fitness())
            # print('self.fitness', self.fitness)

    # 2. #计算个体的适应度
    def evaluate_fitness(self):
        dis = 0
        # print("city_num - 1 = ", city_num - 1)
        # print("***************")
        # print(city_dist_mat)
        # print("self.genes = ", self.genes)
        for i in range(city_num - 1):
            dis += city_dist_mat[self.genes[i]][self.genes[i + 1]]
            # print("he: ", dis)
            if i == city_num - 2:
                # print("adding tail ",self.genes[i + 1], 0, city_dist_mat[self.genes[i + 1]][0])
                dis += city_dist_mat[self.genes[i + 1]][0]  # 回到0
        # print('dis = ', dis)
        if num_person_idx % 20 == 0:
            dis_list.append(dis)
        return 1 / dis


# In[188]:


def copy_list(old):
    new = []
    for element in old:
        new.append(element)
    return new


def sort_win_num(group):
    for i in range(len(group)):
        for j in range(len(group) - i - 1):
            if group[j].fitness < group[j + 1].fitness:
                temp = group[j]
                group[j] = group[j + 1]
                group[j + 1] = temp
    return group


# 定义Ga类
# 3~5，交叉、变异、更新种群，全部在Ga类中实现
class Ga:
    # input_为城市间的距离矩阵
    def __init__(self, input_):
        # 声明一个全局变量
        global city_dist_mat
        city_dist_mat = input_
        # self.best = None
        self.best = Individual(None)
        # 种群
        self.individual_list = []
        # 每一代的最佳个体
        self.result_list = []
        # 每一代个体对应的最佳适应度
        self.fitness_list = []

    # 交叉,这里采用交叉变异
    def cross(self):
        new_gen = []
        # 随机选取一段，含有num_cross个数字（城市）
        num_cross = 3  # 后期可能需要调试的参数,考虑到实际问题里只有5个城市，所以认为3较为合适
        for i in range(0, len(self.individual_list) - 1, 2):
            parent_gen1 = copy_list(self.individual_list[i].genes)
            parent_gen2 = copy_list(self.individual_list[i + 1].genes)
            # 定义一个下表列表
            index_list = [0] * 3
            for i in range(city_num - 3):  # 就是2，即0，1
                index_list[i] = i + 1
            index1_1 = random.choice(index_list)
            index1_2 = index1_1 + 2
            index2_1 = random.choice(index_list)
            index2_2 = index2_1 + 2
            choice_list1 = parent_gen1[index1_1:index1_2 + 1]
            choice_list2 = parent_gen2[index2_1:index2_2 + 1]
            # 利用这一段生成两个子代,下面的赋值只是为了获取长度，所以用哪个父代能可以
            # 也可以直接用city_num直接代替
            son_gen1 = [0] * city_num
            son_gen2 = [0] * city_num
            # 找到之后进行交叉，分别得到son_gen1,son_gen2
            # 先把选中的段复制进去
            son_gen1[index1_1: index1_2 + 1] = choice_list1
            son_gen2[index2_1: index2_2 + 1] = choice_list2
            # 然后左、右“查漏补缺”
            temp1 = choice_list1
            temp2 = choice_list2
            if index1_1 == 0:
                pass
            else:
                for i in range(index1_1):
                    for j in range(city_num):
                        # 如果父代2里面的这个当初没被选中，那就加入son_gene1
                        if parent_gen2[j] not in choice_list1:
                            son_gen1[i] = parent_gen2[j]
                            # 这个时候要扩增choice_list1, 这样parent_gen2里面未被选中的元素才会一个个被遍历到#1
                            choice_list1.append(parent_gen2[j])
                            # 找到之后马上break，防止被覆盖
                            break
            choice_list1 = temp1
            if index1_2 == city_num - 1:
                pass
            else:
                for i in range(index1_2 + 1, city_num):
                    for j in range(city_num):
                        if parent_gen2[j] not in choice_list1:
                            son_gen1[i] = parent_gen2[j]
                            # 这个时候要扩增choice_list1, 这样parent_gen2里面未被选中的元素才会一个个被遍历到#2
                            choice_list1.append(parent_gen2[j])
                            # 找到之后马上break，防止被覆盖
                            break
            # son_gen2亦是如此
            if index2_1 == 0:
                pass
            else:
                for i in range(index2_1):
                    for j in range(city_num):
                        # 如果父代1里面的这个当初没被选中，那就加入son_gen2
                        if parent_gen1[j] not in choice_list2:
                            son_gen2[i] = parent_gen1[j]
                            # 这个时候要扩增choice_list2, 这样parent_gen1里面未被选中的元素才会一个个被遍历到#3
                            choice_list2.append(parent_gen1[j])
                            # 找到之后马上break，防止被覆盖
                            break
            choice_list2 = temp2
            if index2_2 == city_num - 1:
                pass
            else:
                for i in range(index2_2 + 1, city_num):
                    for j in range(city_num):
                        if parent_gen1[j] not in choice_list2:
                            #                             print("i == ", i)
                            son_gen2[i] = parent_gen1[j]
                            # 这个时候要扩增choice_list2, 这样parent_gen1里面未被选中的元素才会一个个被遍历到#4
                            choice_list2.append(parent_gen1[j])
                            # 找到之后马上break，防止被覆盖
                            break
            # 新生成的子代基因加入new_gene列表
            new_gen.append(Individual(son_gen1))
            # print('new_gen[-1].genes', new_gen[-1].genes)
            new_gen.append(Individual(son_gen2))
        return new_gen

    # 变异
    def mutate(self, new_gen):
        change = 0
        mutate_p = 0.02  # 待调参数
        index_list = [0] * (city_num - 1)
        index_1 = 1
        index_2 = 1
        for i in range(city_num - 1):
            index_list[i] = i + 1
        for individual in new_gen:
            if random.random() < mutate_p:
                change += 1
                # 如果变异，采用基于位置的变异,方便起见，直接使用上面定义的index列表
                index_l = random.choice(index_list)
                index_2 = random.choice(index_list)
                while index_1 == index_2:
                    index_2 = random.choice(index_list)
                temp = individual.genes[index_1]
                individual.genes[index_1] = individual.genes[index_2]
                individual.genes[index_2] = temp
        # 变异结束，与老一代的进行合并
        self.individual_list += new_gen

    # 选择
    def select(self):
        # 在此选用锦标赛算法
        # 考虑到5的阶乘是120，所以感觉4个个体一组较为合适,所以group_num = 30,暂定每一代保留60
        group_num = 30  # 待调参数
        group_size = 4
        win_num = 2  # 60/15
        # 锦标赛的胜者列表
        winners = []
        for i in range(group_num):
            # 定义临时列表，存储够一组为止
            group = []
            for j in range(group_size):
                gen_player = random.choice(self.individual_list)
                gen_player = Individual(gen_player.genes)
                group.append(gen_player)
            # 存储完一组之后选出适应度最大的前4个
            group = sort_win_num(group)
            winners += group[: win_num]
        # 选择结束，生成全新一代,赋值给self.individual_list
        self.individual_list = winners

    # 更新种群
    def next_gen(self):
        # 交叉
        new_gene = self.cross()
        # 变异
        self.mutate(new_gene)
        # 选择
        self.select()
        # 获得这一代的最佳个体
        for individual in self.individual_list:
            if individual.fitness > self.best.fitness:
                self.best = individual
    def train(self):
        # 随机出初代种群#
        individual_num = 60
        self.individual_list = [Individual() for _ in range(individual_num)]
        # 迭代
        gen_num = 100
        for i in range(gen_num):
            # 从当代种群中交叉、变异、选择出适应度最佳的个体，获得子代产生新的种群
            self.next_gen()
            # 连接首位
            result = copy.deepcopy(self.best.genes)
            result.append(result[0])
            self.result_list.append(result)
            self.fitness_list.append(self.best.fitness)
        print(self.result_list[-1])
        print('距离总和是：', 1 / self.fitness_list[-1])

    def draw(self):
        x_list = [i for i in range(num_person)]
        y_list = dis_list
        plt.rcParams['figure.figsize'] = (60, 45)
        plt.plot(x_list, y_list, color='g')
        plt.xlabel('Cycles', size=50)
        plt.ylabel('Route', size=50)
        x = np.arange(0, 910, 20)
        y = np.arange(7800, 12000, 100)
        plt.xticks(x)
        plt.yticks(y)
        plt.title('Trends in distance changes', size=50)
        plt.tick_params(labelsize=30)
        plt.savefig("D:\AI_pictures\遗传算法求解TSP问题_1")
        plt.show()


route = Ga(city_dist_mat)
route.train()
route.draw()
