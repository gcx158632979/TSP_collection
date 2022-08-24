import random
import math
import numpy as np
import matplotlib.pyplot as plt
from GA_dubins import calcDubinsPath, Waypoint, dubins_traj
plt.rcParams['font.sans-serif'] = ['KaiTi']
class GA(object):
    def __init__(self, num_city, num_total, iteration, data, waypoint, type):
        self.num_city = num_city
        self.num_total = num_total
        self.scores = []
        self.iteration = iteration
        self.location = data
        self.ga_choose_ratio = 0.2
        self.mutate_ratio = 0.05
        # fruits中存每一个个体是下标的list
        self.dis_mat = self.compute_dis_mat(num_city, data, waypoint, type)
        self.fruits = self.greedy_init(self.dis_mat, num_total, num_city) #初始化每个种群的城市顺序
        # 显示初始化后的最佳路径
        scores = self.compute_adp(self.fruits) #计算种群的适应度
        sort_index = np.argsort(-scores) #按照种群的适应度从大到小排泄
        init_best = self.fruits[sort_index[0]]#选择适应度最大的种群
        init_best = self.location[init_best]#按种群内的排列顺序遍历城市
        # 存储每个iteration的结果，画出收敛图
        self.iter_x = [0]
        self.iter_y = [1. / scores[sort_index[0]]]#被选出的种群的路径长度

    def random_init(self, num_total, num_city):
        tmp = [x for x in range(num_city)]
        result = []
        for i in range(num_total):
            random.shuffle(tmp)
            result.append(tmp.copy())
        return result

    def greedy_init(self, dis_mat, num_total, num_city):
        start_index = 0
        result = []
        for i in range(num_total):
            rest = [x for x in range(0, num_city)]
            # 所有起始点都已经生成了
            if start_index >= num_city:
                start_index = np.random.randint(0, num_city)
                result.append(result[start_index].copy())
                continue
            current = start_index
            rest.remove(current)
            # 找到一条最近邻路径
            result_one = [current]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x

                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result.append(result_one)
            start_index += 1
        return result

    # 计算不同城市之间的距离
    def compute_dis_mat(self, num_city, location, waypoint, type):
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                #print(param.cost)
                if type == "Euclidean":
                    tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
                    dis_mat[i][j] = tmp
                elif type == "Dubins":
                    param = calcDubinsPath(waypoint[i], waypoint[j], 90, 20)
                    dis_mat[i][j] = param.cost
        return dis_mat

    # 计算路径长度
    def compute_pathlen(self, path, dis_mat):#由length的长度变为energy consumption
        try:
            a = path[0]
            b = path[-1]
        except:
            import pdb
            pdb.set_trace()
        result = dis_mat[a][b]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    # 计算种群适应度
    def compute_adp(self, fruits):
        adp = []
        for fruit in fruits:
            if isinstance(fruit, int):
                import pdb
                pdb.set_trace()
            length = self.compute_pathlen(fruit, self.dis_mat)
            adp.append(1.0 / length)
        return np.array(adp)

    def swap_part(self, list1, list2):
        index = len(list1)
        list = list1 + list2
        list = list[::-1]
        return list[:index], list[index:]

    def ga_cross(self, x, y):#待详细看，没看懂
        len_ = len(x)
        assert len(x) == len(y)
        path_list = [t for t in range(len_)]
        order = list(random.sample(path_list, 2))#从0-len()中随机选出两个值
        order.sort()
        start, end = order

        # 找到冲突点并存下他们的下标,x中存储的是y中的下标,y中存储x与它冲突的下标
        tmp = x[start:end]
        x_conflict_index = []
        for sub in tmp:
            index = y.index(sub)#找到x中每个城市的位次在y中的位次
            if not (index >= start and index < end):
                x_conflict_index.append(index)

        y_confict_index = []
        tmp = y[start:end]
        for sub in tmp:
            index = x.index(sub)#找到y中每个城市的位次在x中的位次
            if not (index >= start and index < end):
                y_confict_index.append(index)

        assert len(x_conflict_index) == len(y_confict_index)

        # 交叉
        tmp = x[start:end].copy()
        x[start:end] = y[start:end]
        y[start:end] = tmp

        # 解决冲突
        for index in range(len(x_conflict_index)):
            i = x_conflict_index[index]
            j = y_confict_index[index]
            y[i], x[j] = x[j], y[i]

        assert len(set(x)) == len_ and len(set(y)) == len_
        return list(x), list(y)

    def ga_parent(self, scores, ga_choose_ratio):
        sort_index = np.argsort(-scores).copy()
        sort_index = sort_index[0:int(ga_choose_ratio * len(sort_index))]#选择适应度靠前的种群
        parents = []
        parents_score = []
        for index in sort_index:
            parents.append(self.fruits[index])
            parents_score.append(scores[index])
        return parents, parents_score

    def ga_choose(self, genes_score, genes_choose):
        sum_score = sum(genes_score)
        score_ratio = [sub * 1.0 / sum_score for sub in genes_score]#每个种群的适应度占总种群的比率
        rand1 = np.random.rand()
        rand2 = np.random.rand()#产生一个0-1之间的数
        #可能会出现一个问题，就是一直到循环结束都没有出现index1，index2同时小于0的情况，这个选择方式不太合理。
        for i, sub in enumerate(score_ratio):#适应度高的被选出的概率大
            if rand1 >= 0:
                rand1 -= sub
                if rand1 < 0:
                    index1 = i
            if rand2 >= 0:
                rand2 -= sub
                if rand2 < 0:
                    index2 = i
            if rand1 < 0 and rand2 < 0:#如果index1和index2都被附值了，则循环结束
                break
        return list(genes_choose[index1]), list(genes_choose[index2])

    def ga_mutate(self, gene):
        path_list = [t for t in range(len(gene))]
        order = list(random.sample(path_list, 2))
        start, end = min(order), max(order)
        tmp = gene[start:end]
        # np.random.shuffle(tmp)
        tmp = tmp[::-1]#翻转
        gene[start:end] = tmp
        return list(gene)

    def ga(self):
        # 获得优质父代
        scores = self.compute_adp(self.fruits)
        # 选择部分优秀个体作为父代候选集合
        parents, parents_score = self.ga_parent(scores, self.ga_choose_ratio)
        tmp_best_one = parents[0]
        tmp_best_score = parents_score[0]
        # 新的种群fruits
        fruits = parents.copy()
        # 生成新的种群
        while len(fruits) < self.num_total:
            # 轮盘赌方式对父代进行选择
            gene_x, gene_y = self.ga_choose(parents_score, parents)
            # 交叉
            gene_x_new, gene_y_new = self.ga_cross(gene_x, gene_y)
            # 变异
            if np.random.rand() < self.mutate_ratio:
                gene_x_new = self.ga_mutate(gene_x_new)
            if np.random.rand() < self.mutate_ratio:
                gene_y_new = self.ga_mutate(gene_y_new)
            x_adp = 1. / self.compute_pathlen(gene_x_new, self.dis_mat)
            y_adp = 1. / self.compute_pathlen(gene_y_new, self.dis_mat)
            # 将适应度高的放入种群中
            if x_adp > y_adp and (not gene_x_new in fruits):
                fruits.append(gene_x_new)
            elif x_adp <= y_adp and (not gene_y_new in fruits):
                fruits.append(gene_y_new)

        self.fruits = fruits

        return tmp_best_one, tmp_best_score

    def run(self):
        BEST_LIST = None
        best_score = -math.inf
        self.best_record = []
        for i in range(1, self.iteration + 1):
            tmp_best_one, tmp_best_score = self.ga()#每次ga都只
            self.iter_x.append(i)
            self.iter_y.append(1. / tmp_best_score)
            if tmp_best_score > best_score:
                best_score = tmp_best_score
                BEST_LIST = tmp_best_one
            self.best_record.append(1./best_score)
            print(i,1./best_score)
        print(1./best_score)
        return self.location[BEST_LIST], 1. / best_score


# 读取数据
def read_tsp(path):
    lines = open(path, 'r').readlines()
    assert 'NODE_COORD_SECTION\n' in lines
    index = lines.index('NODE_COORD_SECTION\n')
    data = lines[index + 1:-1]
    tmp = []
    for line in data:
        line = line.strip().split(' ')
        if line[0] == 'EOF':
            continue
        tmpline = []
        for x in line:
            if x == '':
                continue
            else:
                tmpline.append(float(x))
        if tmpline == []:
            continue
        tmp.append(tmpline)
    data = tmp
    return data


data = read_tsp('data/st70.tsp')

data = np.array(data)
data = data[:, 1:]
datalist = []
for i in range(0, len(data)):
    p = [data[i][0], data[i][1]]
    datalist.append(p)
#print(datalist)
#print(datalist.index([64,96]))
Wptz = []
for i in range(0, len(data)):
    p = Waypoint(data[i][0], data[i][1], 0)
    #print(p)
    Wptz.append(p)


'''---------------------------Dubins----------------------------------'''
Best_dubins, Best_path_dubins = math.inf, None

model_dubins = GA(num_city=data.shape[0], num_total=25, iteration=50, data=data.copy(), waypoint=Wptz.copy(), type="Dubins")#num_total代表种群个数
path_dubins, path_len_dubins = model_dubins.run()

if path_len_dubins < Best_dubins:
    Best_dubins = path_len_dubins
    Best_path_dubins = path_dubins

best_order = []
for i in range(0, len(Best_path_dubins)):
    index = datalist.index([Best_path_dubins[i][0],Best_path_dubins[i][1]])
    best_order.append(index)
'''---------------------------Euclidean----------------------------------'''
Best_Euclidean, Best_path_Euclidean = math.inf, None

model_Euclidean = GA(num_city=data.shape[0], num_total=25, iteration=50, data=data.copy(), waypoint=Wptz.copy(), type="Euclidean")#num_total代表种群个数
path_Euclidean, path_len_Euclidean = model_Euclidean.run()
if path_len_Euclidean < Best_Euclidean:
    Best_Euclidean = path_len_Euclidean
    Best_path_Euclidean = path_Euclidean
'''---------------------------Plot Dubins----------------------------------'''    
plt.figure(1)
i = 0
while i<len(best_order)-1:
    param = calcDubinsPath(Wptz[best_order[i]], Wptz[best_order[i+1]], 90, 20)
    path = dubins_traj(param,1)
    # Plot the results
    plt.plot(Wptz[i].x, Wptz[i].y, 'kx')
    plt.plot(Wptz[i + 1].x, Wptz[i + 1].y, 'kx')
    plt.plot(path[:, 0], path[:, 1], 'b-')
    i += 1

# reaching back from the last point to the
# starting one (optional)
param = calcDubinsPath(Wptz[best_order[-1]], Wptz[best_order[0]], 90, 20)
path = dubins_traj(param, 1)

plt.plot(Wptz[-1].x, Wptz[-1].y, 'kx')
plt.plot(Wptz[0].x, Wptz[0].y, 'kx')
plt.plot(path[:, 0], path[:, 1], 'b-')


plt.grid(True)
plt.axis("equal")
plt.title('Dubin\'s Curves Trajectory Generation')
plt.xlabel('X')
plt.ylabel('Y')

'''---------------------------Plot Euclidean----------------------------------'''  
plt.figure(2)
# # 加上一行因为会回到起点
plt.scatter(Best_path_Euclidean[:, 0], Best_path_Euclidean[:,1])
Best_path_Euclidean = np.vstack([Best_path_Euclidean, Best_path_Euclidean[0]])
plt.plot(Best_path_Euclidean[:, 0], Best_path_Euclidean[:, 1])
plt.grid(True)
plt.axis("equal")
plt.title('Euclidean\'s Curves Trajectory Generation')
plt.xlabel('X')
plt.ylabel('Y')
# fig, axs = plt.subplots(2, 1, sharex=False, sharey=False)
# axs[0].scatter(Best_path_Euclidean[:, 0], Best_path_Euclidean[:,1])
# Best_path_Euclidean = np.vstack([Best_path_Euclidean, Best_path_Euclidean[0]])
# axs[0].plot(Best_path_Euclidean[:, 0], Best_path_Euclidean[:, 1])
# axs[0].set_title('规划结果')
# iterations = range(model_Euclidean.iteration)
# best_record = model_Euclidean.best_record
# axs[1].plot(iterations, best_record)
# axs[1].set_title('收敛曲线')
plt.show()


