# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# #线性回归
# #使用numpy生成200个随机点 变成200行1列
# x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
# #生成一些干扰项
# noise=np.random.normal(0,0.2,x_data.shape)
# y_data=np.square(x_data)*noise
#
#
# #定义样本
# #行不确定，但是只有一列
# x=tf.placeholder(tf.float32,[None,1])
# y=tf.placeholder(tf.float32,[None,1])
#
# #构建简单的神经网络的中间层 1个输入神经元  10个中间神经元
# Weight_l1=tf.Variable(tf.random_normal([1,10]))
# baise_l1 = tf.Variable(tf.zeros([1,10]))
# #matmul矩阵的乘法
# Wx_plus_b_11=tf.matmul(x,Weight_l1)+baise_l1
# l1=tf.nn.tanh(Wx_plus_b_11)
#
# #定义输出层  中间层10个神经元 输出层1个神经元
# Weight_12=tf.Variable(tf.random_normal([10,1]))
# biase_12=tf.Variable(tf.zeros([1,1]))
# Wx_plus_b_12=tf.matmul(l1,Weight_12)+biase_12
# prediction=tf.nn.tanh(Wx_plus_b_12)
#
# #二次代价函数
# loss=tf.reduce_mean(tf.square(y-prediction))
#
# #使用梯度下降法
# train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for _ in range(2000):
#         sess.run(train_step,feed_dict={x:x_data,y:y_data})
#
#     #获得预测值
#     predict_value=sess.run(prediction,feed_dict={x:x_data})
#     #画图
#     plt.figure()
#     plt.scatter(x_data,y_data)
#     # r-代表是红色的实线   lw 代表线宽为5
#     plt.plot(x_data,predict_value,'r-',lw=5)
#     plt.show()