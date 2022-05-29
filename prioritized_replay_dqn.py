# -*- coding: UTF-8 -*-
from argparse import Action
from ftplib import B_CRLF
import os
from pickle import NONE

# from sqlalchemy import false, true
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import environment as Env
import tensorflow.compat.v1 as tf
from tensorflow.keras import layers, models
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

class SumTree(object):
    
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame
        # print("p ", self.tree)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        # print("leaf_idx ", leaf_idx, "self.tree[leaf_idx]: ", self.tree[leaf_idx], "self.data[data_idx]: ", self.data[data_idx])
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        # print(self.tree.tree, "size: ", self.tree.tree.shape)
        # print(self.tree.tree[-self.tree.capacity:], "size: ", self.tree.tree[-self.tree.capacity:].shape)
        # print("max_p: ", max_p)
        
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), [], np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        if min_prob == 0:
            min_prob = np.min(self.tree.tree[-self.tree.capacity: -self.tree.capacity+self.tree.data_pointer]) / self.tree.total_p
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            # print("a ", a, "b ", b, "v ", v)
            idx, p, data = self.tree.get_leaf(v)
            # print("idx ", idx, "p ", p, "data ", data)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i] = idx
            b_memory.append(data)
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class DeepQNetwork:
    def __init__(
            self,
            # 输總共多少的relay
            rly_num,
            # 输出多少个action的值
            n_actions,
            # 长宽高等，用feature预测action的值
            n_features,
            # 学习速率
            learning_rate=0.01,
            # reward的折扣值
            reward_decay=0.9,
            # 贪婪算法的值，代表90%的时候选择预测最大的值
            e_greedy=0.9,
            # 隔多少步更换target神经网络的参数变成最新的
            replace_target_iter=300,
            # 记忆库的容量大小
            memory_size=1000,
            # 神经网络学习时一次学习的大小
            batch_size=16,
            # 不断的缩小随机的范围
            e_greedy_increment=0.005,
            output_graph=False,
    ):
        self.rly_num = rly_num
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        # epsilon的最大值
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.01 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        # 记录学习了多少步
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = Memory(capacity=memory_size)

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        # 记录下每步的误差
        self.cost_his = []
        # 记录下每步的reward
        self.reward_his = []
        # 紀錄下每步的age
        self.age_his = []
        # 紀錄下每步random的age
        self.random_age_his = []
        # 紀錄下每步用DBRS走的age
        self.dbrs_age_his = []
        # 紀錄下每步用SARLAT走的age
        self.sarlat_age_his = []
        # 紀錄下每個distence的平均age
        self.age_total_his = []
        # 紀錄下每個distence的平均 dbrs age
        self.dbrs_age_total_his = []
        # 紀錄下每個distence的平均 random age
        self.random_age_total_his = []
        # 紀錄下每個distence的平均 SARLAT age
        self.sarlat_age_total_his = []
        # 紀錄下每個distence使用OR的次數
        self.OR_total_his = []

    def _build_net(self):

        def build_layers(s, n_l1, w_initializer, b_initializer, train):
            with tf.variable_scope('l1'):
                e1 = tf.layers.dense(s, n_l1, tf.nn.relu, kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer, name='e1', trainable = train)
                e1 = tf.layers.flatten(e1)
                out = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                                bias_initializer=b_initializer, name='q', trainable = train)
                return out

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features, self.rly_num], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.variable_scope('eval_net'):
            n_l1, w_initializer, b_initializer = \
                20, tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, n_l1, w_initializer, b_initializer, True)

        with tf.variable_scope('loss'):
            self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)    # for updating Sumtree
            self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
            
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features, self.rly_num], name='s_')    # input
        with tf.variable_scope('target_net'):
            self.q_next = build_layers(self.s_,  n_l1, w_initializer, b_initializer, False)
        
    # 存储记忆
    def store_transition(self, s, a, r, s_):
        transition = [s, a, r, s_]
        # np.array([np.array(s), a, r, np.array(s_)])
        # print(transition)
        self.memory.store(transition)

    # 选择行为
    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = np.reshape(observation, (1,self.n_features,self.rly_num))
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action
    
    # Random选择行为
    def random_choose(self):
        action = np.random.randint(0, self.n_actions)
        return action

    # 选择行为
    def dqn_choose(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = np.reshape(observation, (1,self.n_features,self.rly_num))
        # forward feed the observation and get q value for every actions
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)
        return action
    

    # 学习
    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('target_params_replaced\n')
            
        # sample batch memory from all memory
        
        tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        
        # print("ISWeight ", ISWeights)
        q_next, q_eval = self.sess.run(
                [self.q_next, self.q_eval],
                feed_dict={self.s_: [row[3] for row in batch_memory],
                           self.s: [row[0] for row in batch_memory]})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = [row[1] for row in batch_memory]
        reward = [row[2] for row in batch_memory]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        
        _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                        feed_dict={self.s: [row[0] for row in batch_memory],
                                                self.q_target: q_target,
                                                self.ISWeights: ISWeights})
        self.memory.batch_update(tree_idx, abs_errors)     # update priority
        self.cost_his.append(self.cost)
        
        
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    # 查看学习效果
    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)),
                 self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
    
    def plot_reward(self):
        plt.plot(np.arange(len(self.reward_his)),
                 self.reward_his)
        plt.ylabel('Reward')
        plt.xlabel('training steps')
        plt.show()
    
    def plot_age(self):
        plt.plot(np.arange(len(self.age_total_his)),
                 self.age_total_his)
        plt.plot(np.arange(len(self.dbrs_age_total_his)),
                self.dbrs_age_total_his)
        plt.plot(np.arange(len(self.sarlat_age_total_his)),
                 self.sarlat_age_total_his)
        plt.ylabel('Age')
        plt.xlabel('Steps')
        plt.legend(labels=['DQN', 'DBRS','SAR-LAT'],loc='best')
        plt.show()

    def plot_total_age(self):
        dqn_age = sum(self.age_total_his)/len(self.age_total_his)
        dbrs_age = sum(self.dbrs_age_total_his)/len(self.dbrs_age_total_his)
        sar_let_age = sum(self.sarlat_age_total_his)/len(self.sarlat_age_total_his)
        plt.bar(np.arange(3), [dqn_age, dbrs_age, sar_let_age], color=['red', 'green', 'blue'])
        plt.xticks(np.arange(3), ['DQN', 'DBRS','SAR-LAT'])
        plt.xlabel('Policy')
        plt.ylabel('Age')
        plt.title('Average age for different policy')
        plt.show()

    # def plot_OR(self):
    #     plt.plot(np.arange(len(self.OR_total_his)),
    #              self.OR_total_his)
    #     plt.ylabel('OR usage')
    #     plt.xlabel('Distence')
    #     plt.show()


def run_maze():
    # 控制到第几步学习
    step = 0
    # 进行300回合游戏
    for episode in range(1,1000+1):
        # initial observation
        # 初始化环境，相当于每回合重新开始
        print("Reset.")
        observation = env.reset()
        


        for _ in range(10):
            # fresh env
            # 刷新环境
            # env.render()
            # RL choose action based on observation
            # RL通过观察选择应该选择的动作
            # input("stop!")
            action = RL.choose_action(observation)
            # print("state ",env.returnstate())
            # RL take action and get next observation and reward
            # 环境根据动作，做出反应，主要给出state，reward
            observation_, reward, Age= env.step(action)
            # print("action " ,action, "Reward ", reward, "Age ", Age)
            
            RL.reward_his.append(reward)

            # DQN存储记忆，即（s,a,r,s）
            RL.store_transition(observation, action, reward, observation_)
            
            # 当学习次数大于200，且是5的倍数时才让RL学习
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            # 将下一个state作为本次的state
            observation = observation_

            # 学习的次数
            step += 1
            # input("Step !")
        print("episode: {}/1000".format(episode))


    # input("Testing")

    
    for e in range(1,10+1):
        print("Episode ", e)
        RL.age_his.clear()
        observation = env.reset()
        
        # env.changedis(d)
        for _ in range(2*pow(10,3)):
            # input("pause")
            # print("state ",env.returnstate())
            action = RL.dqn_choose(observation)
            # print("The action by DQN ", action)
            action = env.choose(action)
            # print("So, the action is ", action)
            observation_, reward, Age= env.step(action)
            
            # print("The reward is " , reward, "The age is " , Age)
            
            RL.age_his.append(Age)
            observation = observation_

        RL.age_his = [i for i in RL.age_his if i != -1]

        RL.age_total_his.append(sum(RL.age_his)/len(RL.age_his))
    print(RL.age_total_his)
    print("Average: ", sum(RL.age_total_his)/len(RL.age_total_his))

    for e in range(1,5+1):
        print("Episode ", e)
        RL.dbrs_age_his.clear()
        observation = env.reset()
        # env.changedis(d)
        for _ in range(2*pow(10,3)):
            # input("pause")
            action = env.DBRS_OR()
            # print("So, the action is ", action)
            observation_, reward, Age= env.step(action)

            # print("The age is " , Age)
            RL.dbrs_age_his.append(Age)
            observation = observation_
        RL.dbrs_age_his = [i for i in RL.dbrs_age_his if i != -1]
        RL.dbrs_age_total_his.append(sum(RL.dbrs_age_his)/len(RL.dbrs_age_his))
    print(RL.dbrs_age_total_his)
    print("Average: ", sum(RL.dbrs_age_total_his)/len(RL.dbrs_age_total_his))

    # for d in range(1,5+1):
    #     print("Distense ", d)
    #     RL.random_age_his.clear()
    #     observation = env.reset()
    #     env.changedis(d)
    #     for _ in range(2*pow(10,3)):
    #         # input("pause")
    #         # print("observation ", observation)
    #         action = RL.random_choose()
    #         # print("So, the action is ", action)
    #         observation_, reward, Age= env.step(action)
    #         RL.random_age_his.append(Age)
    #         observation = observation_
    #     RL.random_age_total_his.append(sum(RL.random_age_his)/len(RL.random_age_his))
    # print(RL.random_age_total_his)

    for e in range(1,5+1):
        print("Episode ", e)
        RL.sarlat_age_his.clear()
        observation = env.reset()
        # env.changedis(d)
        for _ in range(2*pow(10,3)):
            # input("pause")
            # print("observation ", observation)
            # print("state ",env.returnstate())
            action = env.SARLAT()
            # print("So, the action is ", action)
            observation_, reward, Age= env.step(action)
            # print("The age is ", Age)
            RL.sarlat_age_his.append(Age)
            observation = observation_
        RL.sarlat_age_his = [i for i in RL.sarlat_age_his if i != -1]
        RL.sarlat_age_total_his.append(sum(RL.sarlat_age_his)/len(RL.sarlat_age_his))
    print(RL.sarlat_age_total_his)
    print("Average: ", sum(RL.sarlat_age_total_his)/len(RL.sarlat_age_total_his))
    print('game over')
    env.destroy()


if __name__ == '__main__':
    env = Env.twohop_relay(3,3,5)
    RL = DeepQNetwork(env.rly_num, env.n_actions, env.n_features,
                      learning_rate=0.001,
                      reward_decay=1.0,
                      e_greedy=0.9,
                      replace_target_iter=5000,
                      memory_size=50000,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()
    RL.plot_age()
    RL.plot_reward()
    RL.plot_total_age()