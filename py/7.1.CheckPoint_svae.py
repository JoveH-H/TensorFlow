import tensorflow as tf

# 定义变量
a = tf.Variable(1, name="a")
b = tf.Variable(2, name="b")

# 定义保存模型
saver = tf.train.Saver()
save_dir = "../save_path/test_model/"

# 定义模型序号
step = 0

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.initialize_all_variables())
    print("v1 =", a.eval())
    print("v2 =", b.eval())
    save_path = saver.save(sess, save_dir + "model", global_step=step)
    print("Model saved in file: ", save_path)
