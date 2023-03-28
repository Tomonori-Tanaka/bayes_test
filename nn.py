import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# データを生成
n=10
x_data = np.random.rand(n).astype(np.float32)
y_data = x_data *  3 + 2

# 　ノイズを加える
y_data = y_data + 0.15 * np.random.randn(n)

# ノイズ付きデータを描画
plt.scatter(x_data,y_data)
plt.show()

W = tf.Variable(tf.zeros([1]))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.compat.v1.train.GradientDescentOptimizer (0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
