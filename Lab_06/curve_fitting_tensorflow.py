import tensorflow as tf
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

mat_file = scio.loadmat('finalMatrix.mat')
data = mat_file.get('finalMatrix')

data=np.asmatrix(data)
ytest = data[:,4]
data = np.delete(data, [0,3,4], 1) 
mean = np.mean(data,0)
std = np.std(data,0)
data = data - mean
data = data / std
N_pacients,N_features=data.shape

#--- initial settings
tf.set_random_seed(1234)#in order to get always the same results
Nsamples=N_pacients
learning_rate = 2e-5

x=tf.placeholder(tf.float32,[Nsamples,N_features])#inputs
t=tf.placeholder(tf.float32,[Nsamples,1])
z=tf.placeholder(tf.float32,[Nsamples,1])#desired outputs

#--- neural netw structure:
w1=tf.Variable(tf.random_normal(shape=[N_features,1], mean=0.0, stddev=1.0, dtype=tf.float32, name="weights"))
b1=tf.Variable(tf.random_normal(shape=[1,1], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases"))
y=tf.matmul(x,w1)+b1

 #--- optimizer structure
cost=tf.reduce_sum(tf.squared_difference(y, t, name="objective_function"))#objective function
optim=tf.train.GradientDescentOptimizer(learning_rate,name="GradientDescent")# use gradient descent in the trainig phase
optim_op = optim.minimize(cost, var_list=[w1,b1])# minimize the objective function changing w1,b1,w2,b2

#--- initialize
init=tf.initialize_all_variables()

#--- run the learning machine
sess = tf.Session()
sess.run(init)

xval=data
tval=ytest

for i in range(50000):
    # train
    train_data={x: xval, t: tval}
    sess.run(optim_op, feed_dict=train_data)
    if i % 1000 == 0:# print the intermediate result
        print(i,cost.eval(feed_dict=train_data,session=sess))
#--- print the final results
print(sess.run(w1),sess.run(b1))

yval=y.eval(feed_dict=train_data,session=sess)

plt.plot(ytest,'r',label='true')
plt.plot(yval,'b',label='estimated')
plt.legend()
plt.title('learning_rate . '+str(learning_rate))
plt.savefig('estimation'+ str(learning_rate)+'.pdf',format='pdf')
plt.show()