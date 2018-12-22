import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
'exec(%matplotlib inline)'
env=gym.make('FrozenLake-v0')
tf.reset_default_graph()
inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4],0,0.01))
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)

nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

# training
init = tf.global_variables_initializer()

# set learning params
y = .99
e = 0.1
num_episodes = 2000

# create list that contain all reward values
jList = []
rList = []
with tf.Session() as sess:
	sess.run(init)
	for i in range(num_episodes):
		# reset enviorment
		s = env.reset()
		rAll = 0
		d = False
		j = 0
		# Q Network
		while j < 99:
			j += 1
			# choose action by greedy approach
			a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
			if np.random.rand(1) < e:
				a[0] = env.action_space.sample()
				# get new state and reward
				s1,r,d,_ = env.step(a[0])
				# obtain q-value by feeding new state 
				Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
				#obtain maxQ
				maxQ1 = np.max(Q1)
				targetQ = allQ
				targetQ[0,a[0]] = r + y*maxQ1
				# train network using target and predicted values 
				_,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
				rAll += r
				s = s1
				if d == True:
					# reduce chance of random action
					e = 1./((i/50) + 10)
					break
		jList.append(j)
		rList.append(rAll)

print "Percent of successful episodes : " + str(sum(rList)/num_episodes) + "%"

