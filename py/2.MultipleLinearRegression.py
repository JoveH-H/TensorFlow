import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle

logdir = 'E:/python/log'  # log存储路径
tf.reset_default_graph()  # 清除default graph和不断增加的节点

# 读取数据文件,并且第一行为数据的开头
df = pd.read_csv("../data/boston housing price.csv", header=0)

# 显示数据摘要描述信息
print(df.describe())

df = np.array(df.values)  # 获取df的值并且转换成 np 的数组格式

y_data = df[:, 12]  # 标签数据

# 归一化
for i in range(12):
    df[:, i] = (df[:, i] - df[:, i].min()) / (df[:, i].max() - df[:, i].min())
x_data = df[:, :12]  # 特征数据

# 定义训练数据的占位符， x是特征值， y是标签值
x = tf.placeholder(tf.float32, [None, 12], name="X")
y = tf.placeholder(tf.float32, [None, 1], name="Y")

# 创建一个命名空间，定义模型函数 y = w1 * x1 + ... + w12 * x12+b
with tf.name_scope("Model1"):
    # 初始化w为shape=（12,1），服从标准差为0.01的随机正态分布的数
    w = tf.Variable(tf.random_normal([12, 1], stddev=0.01), name="W")
    # 初始化b为1.0
    b = tf.Variable(1.0, name="b")


    # 定义模型函数 y = W * X+ b 矩阵相乘matmul
    def model(x, w, b):
        return tf.matmul(x, w) + b


    # 定义线性函数的预测值
    pred = model(x, w, b)

train_epochs = 50  # 迭代次数
learning_rate = 0.01  # 学习率

# 创建一个命名空间，定义损失函数,采用均方差作为损失函数
with tf.name_scope("LossFunction"):
    loss_function = tf.reduce_mean(tf.square(y - pred))

# 梯度下降优化器 设置学习率和优化目标损失最小化
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

sess = tf.Session()  # 建立会话
init = tf.global_variables_initializer()  # 变量初始化
sess.run(init)

# 生成一个写日志的writer，并将当前的TensorFlow计算图写入日志
writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
writer.close()

loss_list = []  # 用于保存loss的值
# 迭代训练
for epoch in range(train_epochs):
    loss_sum = 0.0
    for xs, ys in zip(x_data, y_data):
        xs = xs.reshape(1, 12)
        ys = ys.reshape(1, 1)

        _, loss = sess.run([optimizer, loss_function], feed_dict={x: xs, y: ys})

        loss_sum = loss_sum + loss  # 累加损失

    x_data, y_data = shuffle(x_data, y_data)  # 打乱数据顺序 避免过拟合假性学习

    b0temp = b.eval(session=sess)
    w0temp = w.eval(session=sess)
    loss_average = loss_sum / len(y_data)  # 所有数据的平均损失
    loss_list.append(loss_average)
plt.plot(loss_list)  # 显示迭代过程中的平均代价
plt.show()  # 显示图表

# 随机抽取数据验证
n = np.random.randint(506)
x_test = x_data[n]
x_test = x_test.reshape(1, 12)

predict = sess.run(pred, feed_dict={x: x_test})
print("预测值：%f" % predict)

target = y_data[n]
print("目标值：%f" % target)
