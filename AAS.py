import numpy as np
from matplotlib import pyplot as plt
import random
import copy
from scipy import stats
from math import pi


# Best arm
def best(m):
    Li = np.argsort(m)[-1:]
    Li = np.flipud(Li)
    return Li


# Attack
def Attack():
    if At != 5:
        ret = 0
        for arms in TARGET[At+1]:
            index = int(arms - 1)
            beta = np.sqrt(2 * sigma*sigma * np.log(pi * pi * K * n[index] * n[index] / (3 * delta)) / n[index])
            temp = mu[At] * n[At] + sample - (n[At] + 1) * (mu[index] - 2 * beta - Delta0)
            temp = max(temp, 0)
            if ret < temp:
                ret = temp
    else:
        index = 7
        beta = np.sqrt(0.5 * np.log(pi*pi * K * n[index]*n[index] / (3*delta)) / n[index])
        temp = mu[At] * n[At] + sample - (n[At]+1) * (mu[index] - 2 * beta - Delta0)
        ret = max(temp, 0)
    return ret


Repeat = 10
T = 100000
K = 20
M = 20
Km = 5
Delta0 = 0.05
delta = 0.1
alpha = 10
sigma = 0.1


Mu = [0] * K
index = 0
for i in range(K):
    seed = random.uniform(index, index + 0.1)
    Mu[i] = seed
    index += 0.2
Mu = sorted(Mu, reverse=True)
Mu = np.array(Mu)
print("Mu = " + str(Mu))

MINDELTA = 10
for i in range(K - 1):
    if Mu[i]-Mu[i+1] <= MINDELTA:
        MINDELTA = Mu[i]-Mu[i+1]
print("MinDelta = " + str(MINDELTA))
print()

L = np.ceil((2 * np.log(2*K/delta)) / (MINDELTA * MINDELTA))
print("Learning time: " + str(L))
print()


M1 = [1, 2, 4, 5, 6]
M2 = [1, 3, 5, 7, 9]
M3 = [1, 4, 8, 14, 16]
M4 = [1, 5, 9, 13, 17]

M5 = [2, 3, 7, 11, 15]
M6 = [2, 4, 8, 12, 16]
M7 = [2, 5, 8, 11, 19]
M8 = [2, 6, 12, 16, 20]

M9 = [3, 4, 8, 12, 16]
M10 = [3, 5, 9, 11, 15]
M11 = [3, 6, 12, 16, 18]
M12 = [3, 7, 13, 17, 20]

M13 = [4, 5, 8, 11, 17]
M14 = [4, 6, 9, 15, 17]
M15 = [4, 7, 10, 13, 19]
M16 = [4, 8, 12, 13, 19]

M17 = [5, 6, 10, 14, 18]
M18 = [5, 7, 11, 15, 19]
M19 = [5, 8, 12, 16, 20]

M20 = [6, 8, 14, 16, 18]


# AAS & TAS

Agents = [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15, M16, M17, M18, M19, M20]

AGENTS = copy.deepcopy(Agents)
for i in range(len(AGENTS)):
    for j in range(Km):
        AGENTS[i][j] -= 1

Optimal = 0
for agent in Agents:
    Optimal += Mu[agent[0] - 1]

Dict = dict()
for i in range(K):
    Dict[i+1] = []
for item in Agents:
    for i in range(K):
        if item[0] == (i+1):
            Dict[i + 1].append(Agents.index(item) + 1)
for i in range(K):
    if len(Dict[i+1]) == 0:
        Dict.pop(i+1)
print("Arm:[Agent]: " + str(Dict))

Dict_0 = copy.deepcopy(Dict)
for item in Dict_0:
    Dict_0[item] = len(Dict_0[item])
tuplelist = [(value, key) for key, value in Dict_0.items()]
tuplelist_sorted = sorted(tuplelist, reverse=True)
Order = [value for key, value in tuplelist_sorted]
print("Order of agents: " + str(Order))
print()

Agents_0 = copy.deepcopy(Agents)
temp_arm = []

for arm in Order:
    for agent in Agents_0:
        if arm in agent:
            agent.remove(arm)
            if len(agent) == 0:
                agent.append(arm)
                if arm not in temp_arm:
                    temp_arm.append(arm)
for arm in temp_arm:
    for agent in range(M):
        if arm in Agents[agent] and arm not in Agents_0[agent]:
            Agents_0[agent].append(arm)
for agent in Agents_0:
    agent.sort()
print("After selection: " + str(Agents_0))

for arm in temp_arm:
    Order.remove(arm)
TARGET = dict()
for arm in Order:
    TARGET[arm] = []

for index in range(M):
    for arm in Order:
        if arm in Agents[index]:
            for subarm in Agents[index]:
                if subarm not in Order and subarm not in TARGET[arm]:
                    TARGET[arm].append(subarm)
print("Attack target: " + str(TARGET))
print()


# OA

Regret_OA = np.zeros((Repeat, T))
Cost_OA = np.zeros((Repeat, T))

for counter in range(Repeat):
    print("The " + str(counter) + "'s repeat")

    mu = np.array([Mu[0]+sigma] * K)
    m0 = np.array([Mu[0]+sigma] * K)
    n = np.ones(K)
    r = np.zeros(K)
    UCB = np.zeros(K)

    for t in range(T):

        r = np.sqrt(alpha * np.log(max(t, 1)) / (2 * n))
        UCB = mu + r
        cache_mu = []
        cache_m0 = []
        cache_n = []

        FLAG = [False] * K
        for arms in Order:
            FLAG[arms-1] = True

        for agent in range(M):
            index_temp = best(UCB[AGENTS[agent]])
            At = Agents[agent][int(index_temp)] - 1
            sample = stats.norm.rvs(Mu[At], sigma)
            cache_m0.append(sample)
            cache_n.append(At)

            if (At+1) in Order and FLAG[At]:
                attack = Attack()
                cache_mu.append(sample - attack)
                FLAG[At] = False
            else:
                cache_mu.append(sample)

        temp = 0
        for arm in cache_n:
            mu[arm] = (mu[arm] * n[arm] + cache_mu[temp]) / (n[arm] + 1)
            m0[arm] = (m0[arm] * n[arm] + cache_m0[temp]) / (n[arm] + 1)
            n[arm] += 1
            temp += 1

        for arm in range(K):
            Regret_OA[counter][t] = (Optimal * (t+1) - sum(Mu * n)) / (t+1)
        Cost_OA[counter][t] = sum(cache_m0) - sum(cache_mu)
        if t > 0:
            Cost_OA[counter][t] += Cost_OA[counter][t-1]

std_OA = np.std(Regret_OA, axis=0)
Regret_OA = np.average(Regret_OA, axis=0)
lower_OA, upper_OA = Regret_OA - std_OA, Regret_OA + std_OA

for i in range(Repeat):
    for j in range(T):
        Cost_OA[i][j] /= (j+1)
Cost_std_OA = np.std(Cost_OA, axis=0)
Cost_OA = np.average(Cost_OA, axis=0)
lower_Cost_OA, upper_Cost_OA = Cost_OA - Cost_std_OA, Cost_OA + Cost_std_OA
# print(Regret_OA)
# print(Cost_OA)


# LTA

Regret_LTA = np.zeros((Repeat, T))
Cost_LTA = np.zeros((Repeat, T))

for counter in range(Repeat):
    print("The " + str(counter) + "'s repeat")

    mu = np.array([Mu[0]+3*sigma] * K)
    m0 = np.array([Mu[0]+3*sigma] * K)
    n = np.ones(K)
    r = np.zeros(K)

    LearningFLAG = [i for i in range(K)]

    t = 0
    UCB = mu + r

    while True:

        cache_n = []

        for agent in range(M):
            index_temp = best(UCB[AGENTS[agent]])
            At = Agents[agent][int(index_temp)] - 1
            sample = stats.norm.rvs(Mu[At], sigma)
            cache_n.append(At)
            m0[At] = (m0[At] * n[At] + sample) / (n[At] + 1)
            n[At] += 1

        UCB = mu + np.sqrt(alpha * np.log(t + 1) / (2 * n))
        for i in range(K):
            if UCB[i] > 5:
                UCB[i] = 5
        max_UCB = max(UCB)
        attack = 0
        for arm in cache_n:
            if n[arm] < L:
                if UCB[arm] < max_UCB:
                    attack = (max_UCB - UCB[arm])
                    UCB[arm] = max_UCB
                else:
                    attack = 0
            elif arm in LearningFLAG:
                attack = (mu[arm] - m0[arm])
                LearningFLAG.remove(arm)
                mu[arm] = m0[arm]
                UCB[arm] = m0[arm] + np.sqrt(alpha * np.log(t + 1) / (2 * n[arm]))
            else:
                attack = 0
                mu[arm] = m0[arm]
                UCB[arm] = m0[arm] + np.sqrt(alpha * np.log(t + 1) / (2 * n[arm]))

            Cost_LTA[counter][t] += attack

        for arm in range(K):
            Regret_LTA[counter][t] = (Optimal * (t+1) - sum(Mu * n)) / (t+1)
        if t > 0:
            Cost_LTA[counter][t] += Cost_LTA[counter][t-1]

        t += 1
        if min(n) >= L:
            secondphase = t
            break

    print(secondphase)

    for t in range(secondphase, T):

        r = np.sqrt(alpha * np.log(max(t, 1)) / (2 * n))
        UCB = mu + r
        cache_mu = []
        cache_m0 = []
        cache_n = []

        FLAG = [False] * K
        for arms in Order:
            FLAG[arms-1] = True

        for agent in range(M):
            index_temp = best(UCB[AGENTS[agent]])
            At = Agents[agent][int(index_temp)] - 1
            sample = stats.norm.rvs(Mu[At], sigma)
            cache_m0.append(sample)
            cache_n.append(At)

            if (At+1) in Order and FLAG[At]:
                attack = Attack()
                cache_mu.append(sample - attack)
                Cost_LTA[counter][t] += attack
                FLAG[At] = False
            else:
                cache_mu.append(sample)

        temp = 0
        for arm in cache_n:
            mu[arm] = (mu[arm] * n[arm] + cache_mu[temp]) / (n[arm] + 1)
            m0[arm] = (m0[arm] * n[arm] + cache_m0[temp]) / (n[arm] + 1)
            n[arm] += 1
            temp += 1

        for arm in range(K):
            Regret_LTA[counter][t] = (Optimal * (t+1) - sum(Mu * n)) / (t+1)
        Cost_LTA[counter][t] += Cost_LTA[counter][t-1]

std_LTA = np.std(Regret_LTA, axis=0)
Regret_LTA = np.average(Regret_LTA, axis=0)
lower_LTA, upper_LTA = Regret_LTA - std_LTA, Regret_LTA + std_LTA

for i in range(Repeat):
    for j in range(T):
        Cost_LTA[i][j] /= (j+1)
Cost_std_LTA = np.std(Cost_LTA, axis=0)
Cost_LTA = np.average(Cost_LTA, axis=0)
lower_Cost_LTA, upper_Cost_LTA = Cost_LTA - Cost_std_LTA, Cost_LTA + Cost_std_LTA
# print(Regret_LTA)
# print(Cost_LTA)


# OA w/o AAS

Regret_OAW = np.zeros((Repeat, T))
Cost_OAW = np.zeros((Repeat, T))

for counter in range(Repeat):
    print("The " + str(counter) + "'s repeat")

    mu = np.array([Mu[0]+sigma] * K)
    m0 = np.array([Mu[0]+sigma] * K)
    n = np.ones(K)
    r = np.zeros(K)
    UCB = np.zeros(K)

    for t in range(T):

        r = np.sqrt(alpha * np.log(max(t, 1)) / (2 * n))
        UCB = mu + r
        cache_mu = []
        cache_m0 = []
        cache_n = []

        for agent in range(M):
            index_temp = best(UCB[AGENTS[agent]])
            At = Agents[agent][int(index_temp)] - 1
            sample = stats.norm.rvs(Mu[At], sigma)
            cache_m0.append(sample)
            cache_n.append(At)

            if At == 5:
                attack = Attack()
                cache_mu.append(sample - attack)
            else:
                cache_mu.append(sample)

        temp = 0
        for arm in cache_n:
            mu[arm] = (mu[arm] * n[arm] + cache_mu[temp]) / (n[arm] + 1)
            m0[arm] = (m0[arm] * n[arm] + cache_m0[temp]) / (n[arm] + 1)
            n[arm] += 1
            temp += 1

        for arm in range(K):
            Regret_OAW[counter][t] = (Optimal * (t+1) - sum(Mu * n)) / (t+1)
        Cost_OAW[counter][t] = sum(cache_m0) - sum(cache_mu)
        if t > 0:
            Cost_OAW[counter][t] += Cost_OAW[counter][t-1]

std_OAW = np.std(Regret_OAW, axis=0)
Regret_OAW = np.average(Regret_OAW, axis=0)
lower_OAW, upper_OAW = Regret_OAW - std_OAW, Regret_OAW + std_OAW

for i in range(Repeat):
    for j in range(T):
        Cost_OAW[i][j] /= (j+1)
Cost_std_OAW = np.std(Cost_OAW, axis=0)
Cost_OAW = np.average(Cost_OAW, axis=0)
lower_Cost_OAW, upper_Cost_OAW = Cost_OAW - Cost_std_OAW, Cost_OAW + Cost_std_OAW
# print(Regret_OAW)
# print(Cost_OAW)


# LTA w/o AAS

Regret_OLTA = np.zeros((Repeat, T))
Cost_OLTA = np.zeros((Repeat, T))

for counter in range(Repeat):
    print("The " + str(counter) + "'s repeat")

    mu = np.array([Mu[0]+3*sigma] * K)
    m0 = np.array([Mu[0]+3*sigma] * K)
    n = np.ones(K)
    r = np.zeros(K)

    LearningFLAG = [i for i in range(K)]

    t = 0
    UCB = mu + r

    while True:

        cache_n = []

        for agent in range(M):
            index_temp = best(UCB[AGENTS[agent]])
            At = Agents[agent][int(index_temp)] - 1
            sample = stats.norm.rvs(Mu[At], sigma)
            cache_n.append(At)
            m0[At] = (m0[At] * n[At] + sample) / (n[At] + 1)
            n[At] += 1

        UCB = mu + np.sqrt(alpha * np.log(t + 1) / (2 * n))
        for i in range(K):
            if UCB[i] > 5:
                UCB[i] = 5
        max_UCB = max(UCB)
        attack = 0
        for arm in cache_n:
            if n[arm] < L:
                if UCB[arm] < max_UCB:
                    attack = abs(max_UCB - UCB[arm])
                    UCB[arm] = max_UCB
                else:
                    attack = 0
            elif arm in LearningFLAG:
                attack = mu[arm] - m0[arm]
                LearningFLAG.remove(arm)
                mu[arm] = m0[arm]
                UCB[arm] = m0[arm] + np.sqrt(alpha * np.log(t + 1) / (2 * n[arm]))
            else:
                attack = 0
                mu[arm] = m0[arm]
                UCB[arm] = m0[arm] + np.sqrt(alpha * np.log(t + 1) / (2 * n[arm]))

            Cost_OLTA[counter][t] += attack

        for arm in range(K):
            Regret_OLTA[counter][t] = (Optimal * (t+1) - sum(Mu * n)) / (t+1)
        if t > 0:
            Cost_OLTA[counter][t] += Cost_OLTA[counter][t-1]

        t += 1
        if min(n) >= L:
            secondphase = t
            break

    print(secondphase)

    for t in range(secondphase, T):

        r = np.sqrt(alpha * np.log(max(t, 1)) / (2 * n))
        UCB = mu + r
        cache_mu = []
        cache_m0 = []
        cache_n = []

        for agent in range(M):
            index_temp = best(UCB[AGENTS[agent]])
            At = Agents[agent][int(index_temp)] - 1
            sample = stats.norm.rvs(Mu[At], sigma)
            cache_m0.append(sample)
            cache_n.append(At)

            if At == 5:
                attack = Attack()
                cache_mu.append(sample - attack)
                Cost_OLTA[counter][t] += attack
            else:
                cache_mu.append(sample)

        temp = 0
        for arm in cache_n:
            mu[arm] = (mu[arm] * n[arm] + cache_mu[temp]) / (n[arm] + 1)
            m0[arm] = (m0[arm] * n[arm] + cache_m0[temp]) / (n[arm] + 1)
            n[arm] += 1
            temp += 1

        for arm in range(K):
            Regret_OLTA[counter][t] = (Optimal * (t+1) - sum(Mu * n)) / (t+1)
        Cost_OLTA[counter][t] += Cost_OLTA[counter][t-1]

std_OLTA = np.std(Regret_OLTA, axis=0)
Regret_OLTA = np.average(Regret_OLTA, axis=0)
lower_OLTA, upper_OLTA = Regret_OLTA - std_OLTA, Regret_OLTA + std_OLTA

for i in range(Repeat):
    for j in range(T):
        Cost_OLTA[i][j] /= (j+1)
Cost_std_OLTA = np.std(Cost_OLTA, axis=0)
Cost_OLTA = np.average(Cost_OLTA, axis=0)
lower_Cost_OLTA, upper_Cost_OLTA = Cost_OLTA - Cost_std_OLTA, Cost_OLTA + Cost_std_OLTA
# print(Regret_OLTA)
# print(Cost_OLTA)


# No Attack

Regret = np.zeros((Repeat, T))

for counter in range(Repeat):
    print("The " + str(counter) + "'s repeat")

    mu = np.array([Mu[0]+sigma] * K)
    n = np.ones(K)
    r = np.zeros(K)
    UCB = np.zeros(K)

    for t in range(T):

        r = np.sqrt(alpha * np.log(max(t, 1)) / (2 * n))
        UCB, LCB = mu + r, mu - r
        cache_mu = []
        cache_n = []

        for agent in range(M):
            index_temp = best(UCB[AGENTS[agent]])
            At = Agents[agent][int(index_temp)] - 1
            sample = stats.norm.rvs(Mu[At], sigma)
            cache_mu.append(sample)
            cache_n.append(At)

        temp = 0
        for arm in cache_n:
            mu[arm] = (mu[arm] * n[arm] + cache_mu[temp]) / (n[arm] + 1)
            n[arm] += 1
            temp += 1

        for arm in range(K):
            Regret[counter][t] = (Optimal * (t+1) - sum(Mu * n)) / (t+1)

std = np.std(Regret, axis=0)
Regret = np.average(Regret, axis=0)
lower, upper = Regret - std, Regret + std





plt.figure()
plt.grid(True)
x = np.linspace(0, T, T)

plt.plot(x, Regret_LTA, label="LTA", color="red", linewidth="1.2")
plt.fill_between(x, lower_LTA, upper_LTA, alpha=0.5, color='pink')
plt.plot(x, Regret_OA, label="OA", color="orange", linewidth="1.2", linestyle="--")
plt.fill_between(x, lower_OA, upper_OA, alpha=0.5, color='gold')
plt.plot(x, Regret_OLTA, label="LTA w/o AAS", color="blue", linewidth="1.2", linestyle=":")
plt.fill_between(x, lower_OLTA, upper_OLTA, alpha=0.5, color='deepskyblue')
plt.plot(x, Regret_OAW, label="OA w/o AAS", color="purple", linewidth="1.2", linestyle="-.")
plt.fill_between(x, lower_OAW, upper_OAW, alpha=0.5, color='mediumpurple')
plt.plot(x, Regret, label="No attack", color="green", linewidth="1.2")
plt.fill_between(x, lower, upper, alpha=0.5, color='lime')

plt.xlabel("Round", fontsize=20)
plt.ylabel("Average Regret", fontsize=20)
plt.legend(fontsize=12)
plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("regret.png", dpi=600, bbox_inches='tight')
plt.show()


plt.figure()
plt.grid(True)
x = np.linspace(0, T, T)

plt.plot(x, Cost_LTA, label="LTA", color="red", linewidth="1.2")
plt.fill_between(x, lower_Cost_LTA, upper_Cost_LTA, alpha=0.5, color='pink')
plt.plot(x, Cost_OA, label="OA", color="orange", linewidth="1.2", linestyle="--")
plt.fill_between(x, lower_Cost_OA, upper_Cost_OA, alpha=0.5, color='gold')
plt.plot(x, Cost_OLTA, label="LTA w/o AAS", color="blue", linewidth="1.2", linestyle=":")
plt.fill_between(x, lower_Cost_OLTA, upper_Cost_OLTA, alpha=0.5, color='deepskyblue')
plt.plot(x, Cost_OAW, label="OA w/o AAS", color="purple", linewidth="1.2", linestyle="-.")
plt.fill_between(x, lower_Cost_OAW, upper_Cost_OAW, alpha=0.5, color='mediumpurple')

plt.xlabel("Round", fontsize=20)
plt.ylabel("Average Cost", fontsize=20)
plt.legend(fontsize=12)
plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("cost.png", dpi=600, bbox_inches='tight')
plt.show()
