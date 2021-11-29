import argparse
import os
import time
from OU import OU
from drone_env import drone_env_block
import datetime
import matplotlib.pyplot as plt
import math
import random
import numpy as np
import tensorflow as tf
from algorithms.ReplayMemory import ReplayMemory
import tensorlayer as tl
from algorithms.OUNoise import OrnsteinUhlenbeckActionNoise
from tensorflow.keras.layers import TimeDistributed, BatchNormalization, Flatten, Add, Lambda, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, GRU, Input, ELU, Activation, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true',  default=False)
parser.add_argument('--seqsize', type=int, default=8)
args = parser.parse_args()
wait_before_train = 4
memory_size = 200000
total_step = 0
train_indicator = 1
OU = OU()

class DDPG(object):
    def __init__(self):
        self.memory = ReplayMemory(memory_size)
        self.pointer = 0
    def store_transition(self, s, a, r, s_, terminal):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        # 整理s，s_,方便直接输入网络计算
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)

        # 把s, a, [r], s_横向堆叠
        # transition = np.hstack((s, a, [r], s_))
        self.memory.append(s, a, r, s_, terminal)
        # pointer是记录了曾经有多少数据进来。
        # index是记录当前最新进来的数据位置。
        # 所以是一个循环，当MEMORY_CAPACITY满了以后，index就重新在最底开始了
        # index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        # # 把transition，也就是s, a, [r], s_存进去。
        # self.memory[index, :] = transition
        self.pointer += 1


class Actor:
   def __init__(self, state_dim, action_dim, action_bound, learning_rate, batch, tau=0.01):
       self.action_dim = action_dim
       self.action_bound = action_bound
       self.state_dim = state_dim
       self.lr = learning_rate
       self.batch = batch
       self.tau = tau
       self.target = self._build_net(self.state_dim, trainable=False, scope='target')
       self.eval = self._build_net(self.state_dim, trainable=True, scope='eval')
       self.target.set_weights(self.eval.get_weights())
       self.metric = tf.keras.metrics.MeanSquaredError()
       self.metric.reset_states()
       self.action_noise = OrnsteinUhlenbeckActionNoise(np.array([0.5, 0, 0]))
   def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')

        self.eval.save_weights('model/ddpg_actor.hdf5')
        self.target.save_weights('model/ddpg_actor_target.hdf5')

   def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        self.eval.load_weights('model/ddpg_actor.hdf5')
        self.target.load_weights('model/ddpg_actor_target.hdf5')

   def _build_net(self,input_state_shape, trainable=True, scope=''):
       S = Input(shape=input_state_shape, name='state')
       image = BatchNormalization()(S)
       image = TimeDistributed(
           Conv2D(32, (3, 3), activation=tf.nn.relu, kernel_initializer=tf.initializers.he_normal()))(image)
       image = TimeDistributed(MaxPooling2D((3, 3)))(image)
       # image = TimeDistributed(Dropout(keep=0.8))(image)
       image = TimeDistributed(
           Conv2D(32, (3, 3), activation=tf.nn.relu, kernel_initializer=tf.initializers.he_normal()))(image)
       image = TimeDistributed(MaxPooling2D((3, 3)))(image)
       # image = TimeDistributed(Dropout(keep=0.8))(image)
       image = TimeDistributed(
           Conv2D(32, (3, 3), activation=tf.nn.relu, kernel_initializer=tf.initializers.he_normal()))(image)
       image = TimeDistributed(MaxPooling2D((3, 3)))(image)

       image = TimeDistributed(Flatten())(image)
       image = GRU(48, kernel_initializer='he_normal', use_bias=False)(image)
       image = BatchNormalization()(image)
       image = Activation('tanh')(image)

       policy = Dense(32, activation=tf.nn.relu)(image)
       policy = BatchNormalization()(policy)
       policy = Dense(32, activation=tf.nn.relu)(policy)
       policy = BatchNormalization()(policy)

       V = Dense(self.action_dim, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                 activation=tf.nn.tanh)(policy)
       V = Lambda(lambda x: K.clip(x, -self.action_bound, self.action_bound))(V)
       model = Model(inputs=S, outputs=V)
       return model

   def choose_action(self, s, wait_before_train, step, epsilon_model):
       """
       Choose action
       :param s: state
       :return: act
       """
       epsilon = 1
       noise = True
       action_hs = np.array(self.eval.predict(s)[0])
       action = np.zeros([3], dtype="float32")
       noise_t = np.zeros([action_dim])
       # noise_t = np.random.normal(0, 1, a_dim)
       # noise_t = np.array(noise_t, dtype=np.float32)
       # action = np.clip(action + noise, -1, 1)
       action_truth = np.zeros([3])
       action[0] = 0.5
       action[1] = action_hs[0]
       action[2] = action_hs[1]
       epsilon_ceil = 0.95
       if epsilon_model == 'test':
           noise = False
       elif epsilon_model == 'linear':
           epsilon = epsilon_ceil * (step - wait_before_train) / (20000 - wait_before_train)
           if epsilon > epsilon_ceil:
               epsilon = epsilon_ceil
       elif epsilon_model == 'exponential':
           epsilon = 1 - math.exp(-2 / (100000 - wait_before_train) * (step - wait_before_train))
           if epsilon > epsilon_ceil:
               epsilon = epsilon_ceil
       if epsilon_model != 'test':
           if random.random() > epsilon:
               noise = True
           else:
               noise = False
       if noise:
           action_truth[0] = action[0]
           action_truth[1] = action[1]
           action_truth[2] = action[2]

           # noise_t[0] =train_indicator * OU.function(action[0], 0.5, 1.00, 0.10)
           noise_t[0] = train_indicator * OU.function(action[1], 0.0, 0.6, 0.30)
           noise_t[1] = train_indicator * OU.function(action[2], 0.0, 0.6, 0.30)

           # action[0] = np.clip(action[0]+noise_t[0], -self.a_bound, self.a_bound)
           action[1] = np.clip(action[1] + noise_t[0], -self.action_bound, self.action_bound)
           action[2] = np.clip(action[2] + noise_t[1], -self.action_bound, self.action_bound)


       else:
           # for i in range(3):
           #     action[i] = np.clip(action[i], -self.a_bound, self.a_bound)
           action_truth[0] = action[0]
           action_truth[1] = action[1]
           action_truth[2] = action[2]

           # noise_t[0] =train_indicator * OU.function(action[0], 0.5, 1.00, 0.10)
           noise_t[0] = train_indicator * OU.function(action[1], 0.0, 0.06, 0.30)
           noise_t[1] = train_indicator * OU.function(action[2], 0.0, 0.06, 0.30)

           # action[0] = np.clip(action[0]+noise_t[0], -self.a_bound, self.a_bound)
           action[1] = np.clip(action[1] + noise_t[0], -self.a_bound, self.a_bound)
           action[2] = np.clip(action[2] + noise_t[1], -self.a_bound, self.a_bound)
       # self.num_action_taken += 1

       return action, noise_t, action_truth

   def learn(self, s, g):
       self.metric.reset_states()
       with tf.GradientTape() as tape:
           y = self.eval(s)
           loss = tf.multiply(-g,y)
           loss = tf.reduce_mean(loss)
       grads = tape.gradient(loss, self.eval.trainable_weights)  # dq/da *da/d(\theta)
       grads_and_vars = zip(grads, self.eval.trainable_weights)
       # opt = tf.keras.optimizers.SGD(lr=self.lr,momentum=1e-2,nesterov=1e-2)
       opt = tf.keras.optimizers.Adagrad(lr=self.lr)
       opt.apply_gradients(grads_and_vars)
       target_parmas = [(1-self.tau) * t + self.tau*e for t, e in zip(self.target.get_weights(), self.eval.get_weights())]
       self.target.set_weights(target_parmas)
       return loss
class Critic:
   def __init__(self, state_dim, action_dim, learnig_rate, gamma, tau, a_target,a_eval):
       self.state_dim = state_dim
       self.action_dim = action_dim
       self.lr = learnig_rate
       self.gamma = gamma
       self.tau = tau
       self.a = a_eval  # Actor.eval
       self.a_ = a_target #Actor.target
       self.q = self._build_net(self.state_dim, self.action_dim, trainable=True, scope='eval')
       self.q_ = self._build_net(self.state_dim, self.action_dim,trainable=False, scope='target')
       self.q_.set_weights(self.q.get_weights())
       self.metric = tf.keras.metrics.MeanSquaredError()
       self.metric.reset_states()
   # @tf.function
   def save_ckpt(self):
       """
       save trained weights
       :return: None
       """
       if not os.path.exists('model'):
           os.makedirs('model')

       self.q.save_weights('model/ddpg_critic.hdf5')
       self.q_.save_weights('model/ddpg_critic_target.hdf5')

   def load_ckpt(self):
       """
       load trained weights
       :return: None
       """
       self.q.load_weights('model/ddpg_critic.hdf5')
       self.q_.load_weights('model/ddpg_critic_target.hdf5')
   def _build_net(self, input_state_shape, input_action_shape, trainable=True, scope=''):
       S = Input(input_state_shape, name='state')
       image = BatchNormalization()(S)
       image = TimeDistributed(
           Conv2D(32, (3, 3), activation=tf.nn.relu, kernel_initializer=tf.initializers.he_normal()))(S)
       image = TimeDistributed(MaxPooling2D((3, 3)))(image)
       # image = TimeDistributed(Dropout(keep=0.8))(image)
       image = TimeDistributed(
           Conv2D(32, (3, 3), activation=tf.nn.relu, kernel_initializer=tf.initializers.he_normal()))(image)
       image = TimeDistributed(MaxPooling2D((3, 3)))(image)
       # image = TimeDistributed(Dropout(keep=0.8))(image)
       image = TimeDistributed(
           Conv2D(32, (3, 3), activation=tf.nn.relu, kernel_initializer=tf.initializers.he_normal()))(image)
       image = TimeDistributed(MaxPooling2D((3, 3)))(image)

       image = TimeDistributed(Flatten())(image)
       image = GRU(48, kernel_initializer='he_normal', use_bias=False)(image)
       image = BatchNormalization()(image)
       image = Activation('tanh')(image)

       state = Dense(32, activation=tf.nn.relu)(image)

       a = Input([input_action_shape], name='C_a_input')
       x = tf.keras.layers.concatenate([state, a])
       Q = Dense(32 + self.action_dim, activation=tf.nn.relu)(x)
       Q = BatchNormalization()(Q)
       Q = Dense(1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(Q)
       model = tf.keras.models.Model(inputs=[S, a], outputs=Q)
       return model
   # @tf.function
   def learn(self, s, a, r, s_):
       self.metric.reset_states()
       with tf.GradientTape() as tape:
           pre = self.q([s, a])
           a_ = self.a_.predict(s_)
           y = r + self.gamma*self.q_([s_, a_])  # y_i = r_i + \gamma * r_{i+1}
           loss = tf.reduce_mean(tf.keras.losses.Huber()(y, pre))
           pred = y
       grads = tape.gradient(loss, self.q.trainable_variables)
       grads_and_vars = zip(grads, self.q.trainable_variables)
       opt = tf.keras.optimizers.Adam(lr=self.lr)
       opt.apply_gradients(grads_and_vars)
       eval_parmas = self.q.get_weights()
       target_parmas = self.q_.get_weights()
       # bug here, fixed
       target_parmas = [(1-self.tau)*t + self.tau*e for t, e in zip(target_parmas, eval_parmas)]
       self.q_.set_weights(target_parmas)
       with tf.GradientTape() as tape1:
           a = self.a(s)
           y = self.q([s, a])
       grades = tape1.gradient(y, a)  # dq/da
       return grades, loss, pre, pred


if __name__ == '__main__':
   physical_devices = tf.config.experimental.list_physical_devices('GPU')
   if len(physical_devices) > 0:
       for k in range(len(physical_devices)):
           tf.config.experimental.set_memory_growth(physical_devices[k], True)
           # tf.config.experimental
           print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
   else:
       print("Not enough GPU hardware devices available")
   #tf.keras.backend.set_floatx('float64')
   np.random.seed(0)
   tf.random.set_seed(0)
   l_a = 1e-3
   l_c = 1e-3
   state_dim = [8, 144, 256, 1]
   action_dim = 2
   action_bound = 1
   batch = 4
   gamma = 0.9
   tau = 0.001
   actor = Actor(state_dim, action_dim, action_bound, l_a, batch)

   critic = Critic(state_dim, action_dim, l_c, gamma, tau, actor.eval, actor.target)


   env = drone_env_block(aim=[58, 125, 10])
   max_episodes = 100
   max_ep_steps = 2000
   isstart = 1
   R = []
   net = []
   ddpg = DDPG()
   if args.train:  # train
       current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
       train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
       train_summary_writer = tf.summary.create_file_writer(train_log_dir)
       reward_buffer = []  # 用于记录每个EP的reward，统计变化
       t0 = time.time()  # 统计时间
       epsilon_model = 'linear'
       s = env.reset()
       state = np.stack([s] * args.seqsize, axis=1)
       sh = []
       e, success, ep_reward, step_count = 0, 0, 0, 0
       while True:
           step_count+= 1
           a, noise, action_truth = actor.choose_action(state, wait_before_train, step_count, epsilon_model)
           print("Action: {} noise: {} Action After Noise: {} pointer: {}".format(action_truth, noise, a, ddpg.pointer))

           s_, r, done, info = env.step(a)
           next_state = np.append(state[:, 1:], [s_], axis=1)
           action_pre = np.zeros([action_dim], dtype='float32')
           action_pre[0] = a[1]
           action_pre[1] = a[2]
           ddpg.store_transition(state, action_pre, r, next_state, done)
           if ddpg.pointer > wait_before_train:
               if isstart == 1:
                   print('start train')
                   isstart += 1
               bs, ba, br, bs_, terminals= ddpg.memory.sample(batch)
               grades, loss, pre, pred = critic.learn(bs, ba, br, bs_)
               a_loss = actor.learn(bs, grades)
               critic.a = actor.eval
               critic.a_ = actor.target
               # print(env.shares_held)
           state = next_state
           ep_reward += r  # 记录当前EP的总reward
           step_count += 1
           total_step += 1
           if done:
               if info == "success":
                   success += 1
               # print(
               #     '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
               #         i, MAX_EPISODES, ep_reward,
               #         time.time() - t1
               #     ), end=''
               # )
               print(" " * 80, end="\r")

               print("episode {} finish, average reward: {}, total success: {} result: {} step: {}".format(e,
                                                                                                           ep_reward / step_count,
                                                                                                           success,
                                                                                                           info,
                                                                                                           step_count).ljust(
                   80, " "))
               actor.save_ckpt()
               critic.save_ckpt()
               with train_summary_writer.as_default():  # 将loss写入TensorBoard
                   tf.summary.scalar('Episode Reward', ep_reward, step=e)
               with train_summary_writer.as_default():  # 将loss写入TensorBoard
                   tf.summary.scalar('Episode Reward Step', ep_reward, step=total_step)
               ep_reward = 0
               step_count = 0

               e += 1
               if e % 10 == 0:
                   actor.save_ckpt()
                   critic.save_ckpt()
               s = env.reset()
               state = np.stack([s] * args.seqsize, axis=1)
           # if j == max_ep_steps - 1:
           #     print('Episode:', i, ' Reward: %f' % ep_reward, 'Explore: %.2f' % var, 'net_worth:%.2f' % env.net)
           #     net.append(env.net)
           #     R.append(ep_reward)
           #     data = env.get_data(ts_code=env.ts_code, start_date=env.trade_date[0], limit_num=len(sh))['close']
           #     plt.close('all')
           #     fig = plt.figure()
           #     ax = fig.add_subplot(1, 1, 1)
           #     ax.plot(range(len(data)), data, 'r', label='price')
           #     ax.legend(loc='best')
           #     ax.set_xticks([ix for ix in range(len(data)) if ix % 30 == 0])
           #     ax.set_xticklabels([ii for ix, ii in enumerate(env.trade_date[:len(data)]) if ix % 30 == 0], \
           #                        rotation=60)
           #     ax2 = ax.twinx()
           #     ax2.bar(range(len(sh)), sh, label='shares held')
           #     ax2.legend(loc='upper left')
           #     ax2.set_ylabel('shares')
           #     ax.set_ylabel('$yuan$')
           #     plt.xlabel('trade_date')
           #     plt.savefig(f'./10-10-NO.{i}.png')
           #     # plt.show()