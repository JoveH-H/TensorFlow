import tensorflow as tf

# 定义变量
a = tf.Variable(0, name="a")
b = tf.Variable(0, name="b")

# 定义保存模型
saver = tf.train.Saver()
save_dir = "../save_path/test_model/"

# 定义模型序号
step = 0

with tf.Session() as sess:
    # 恢复保存模型
    # 如果有检查点文件, 读取最新的检查点文件，恢复各种变量值
    ckpt_dir = tf.train.latest_checkpoint(save_dir)
    if ckpt_dir != None:
        saver.restore(sess, ckpt_dir)
    else:
        # 变量初始化
        sess.run(tf.initialize_all_variables())
    # 或者直接读取
    # saver.restore(sess, save_dir + "model-{}".format(step))
    print("v1 =", a.eval())
    print("v2 =", b.eval())
