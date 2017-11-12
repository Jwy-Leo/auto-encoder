import tensorflow as tf
import numpy as np 
import cv2 
from tensorflow.examples.tutorials.mnist import input_data
def activate_function(input_data,activiate):
	return activiate(input_data)
lr=0.01
epoch=1000
batch_size=256
disp_iter=1
LAY_NUM=[784,256,128,256,784]
mnist=input_data.read_data_sets("/tmp/data/",one_hot=False)
Auto_encod_w=[ tf.Variable(tf.random_normal([LAY_NUM[_],LAY_NUM[_+1]])) for _ in range(len(LAY_NUM)-1)]
Auto_encod_b=[tf.Variable(tf.random_normal([LAY_NUM[_+1]])) for _ in range(len(LAY_NUM)-1)]

X=tf.placeholder(shape=[None,LAY_NUM[0]],dtype=tf.float32)
In=X
for i in range(2):
	In=activate_function(tf.add(tf.matmul(In,Auto_encod_w[i]),Auto_encod_b[i]),tf.nn.tanh)
for i in range(2):
	In=activate_function(tf.add(tf.matmul(In,Auto_encod_w[i+2]),Auto_encod_b[i+2]),tf.nn.tanh)

cost=tf.reduce_mean(tf.pow(X-(In+1)/2,2))
optim=tf.train.AdamOptimizer(lr).minimize(cost)
tf.summary.scalar('cost',cost)
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter("logs/",tf.Session().graph)
with tf.Session() as sess:
	init=tf.global_variables_initializer()
	sess.run(init)
	total_batch=int (mnist.train.num_examples/batch_size)
	for epoch_ind in range (epoch):
		for i in range (total_batch):
			batchX,batchY=mnist.train.next_batch(batch_size)
			_,c,write=sess.run([optim,cost,merged],feed_dict={X:batchX})
			writer.add_summary(write)
		if epoch_ind%disp_iter==0:
			print("Epoch:%04d\t"%(epoch_ind+1),"cost=","{:.9f}".format(c))
	writer.close()
	ED=sess.run(In,feed_dict={X:mnist.test.images[:10]})
	for i in range(10):
		cv2.imwrite(str(i)+"real.jpg",np.reshape(mnist.test.images[i],(28,28))*255)
		cv2.imwrite(str(i)+"Gen.jpg",(np.reshape(ED[i],(28,28))+1)*127)
