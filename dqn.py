# -*- coding: UTF-8 -*-
import os

# from sqlalchemy import false, true
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import environment as Env
import tensorflow.compat.v1 as tf
from tensorflow.keras import layers, models
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt


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
            memory_size=10000,
            # 神经网络学习时一次学习的大小
            batch_size=16,
            # 不断的缩小随机的范围
            e_greedy_increment=0.01,
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
        self.epsilon = 0.05 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        # 记录学习了多少步
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = []

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
        print(self.rly_num)
        self.s = tf.placeholder(tf.float32, [None,self.n_features, self.rly_num], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None,self.n_features, self.rly_num], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)


        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='e1')
            e1 = tf.layers.flatten(e1)
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                            bias_initializer=b_initializer, name='q')
            self.eval_weights = tf.get_default_graph().get_tensor_by_name(os.path.split(self.q_eval.name)[0] + '/kernel:0')
            self.eval_bias = tf.get_default_graph().get_tensor_by_name( os.path.split(self.q_eval.name)[0] + '/bias:0')
            
            print("q_eval ", self.q_eval)
            print(self.eval_weights)

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='t1')
            t1 = tf.layers.flatten(t1)
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                            bias_initializer=b_initializer, name='t2')
            self.tar_weights = tf.get_default_graph().get_tensor_by_name(os.path.split(self.q_next.name)[0] + '/kernel:0')
            self.tar_bias = tf.get_default_graph().get_tensor_by_name( os.path.split(self.q_next.name)[0] + '/bias:0')
        
        with tf.variable_scope('q_target'):
            print(self.r)
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            print(q_target)
            self.q_target = tf.stop_gradient(q_target)
            print(self.q_target)
            

        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
            
        with tf.variable_scope('loss'):
            print(self.q_target)
            print(self.q_eval_wrt_a)
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
            print(self.loss)
            
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        
    # 存储记忆
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        
        transition = [s, a, r, s_]
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        if(len(self.memory)>index):
            self.memory[index] = transition
        else:
            self.memory.append(transition)
        self.memory_counter += 1

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
        # print("my action", action)
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
            t = self.sess.run(self.target_replace_op)
            print('target_params_replaced\n')
            
        # sample batch memory from all memory
        
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = [self.memory[i] for i in sample_index]
        
        # if self.learn_step_counter == 10 or self.learn_step_counter == 0 or self.learn_step_counter == 100 or self.learn_step_counter == 199 or self.learn_step_counter == 200 or self.learn_step_counter == 201:
        # evalqq = self.sess.run(self.eval_weights, feed_dict={self.s: [row[0] for row in batch_memory]})
        # print("evalqq",evalqq)
        # tarqq = self.sess.run(self.tar_weights, feed_dict={self.s_: [row[0] for row in batch_memory]})
        # print("tarqq",tarqq)
        # q_eval = self.sess.run(self.q_eval, feed_dict={self.s: [row[0] for row in batch_memory]})
        # print("q_eval",q_eval)
        # input("stop")
        
        _, cost = self.sess.run(
                    [self._train_op, self.loss],
                    feed_dict={
                        self.s: [row[0] for row in batch_memory],
                        self.a: [row[1] for row in batch_memory],
                        self.r: [row[2] for row in batch_memory],
                        self.s_: [row[3] for row in batch_memory],
                    })
        # print("cost: ",cost)
        self.cost_his.append(cost)
        # input("stop")
        
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
        # plt.plot(np.arange(len(self.random_age_total_his)),
        #         self.random_age_total_his)
        plt.plot(np.arange(len(self.dbrs_age_total_his)),
                self.dbrs_age_total_his)
        plt.plot(np.arange(len(self.sarlat_age_total_his)),
                 self.sarlat_age_total_his)
        plt.ylabel('Age')
        plt.xlabel('Steps')
        plt.legend(labels=['DQN','DBRS','SAR-LAT'],loc='best')
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
                      reward_decay=1,
                      e_greedy=0.9,
                      replace_target_iter=5000,
                      memory_size=50000,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()
    RL.plot_age()
    # RL.plot_OR()