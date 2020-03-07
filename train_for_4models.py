from cnn_utils import *
from data_util import *
import numpy as np
import tensorflow as tf

"""
CAPTC_Identification_mulit-models_mulit-labels
输入图像:60*160*3>>>60*160*1
标签:'2q2D'
一次训练出4个model,每个模型负责预测一个标签 如model01预测'2',model02预测'q',model03预测'2',model01预测'D'
"""

def cnn(input,keep_prob=0.75,
                LABEL_DEPTH=62,
                is_train=True
                ):
    """
    :param input: 输入图像的尺寸[-1,60,160,1]
    :param keep_prob: 每个元素被保留的概率
    :param MAX_CAPTCHA: 验证码共有4个字符
    :param CHAR_SET_LEN: 每个字符有62种选择,从A-Z,a-z,0-9
    :return:
    """
    # x = tf.reshape(tf.cast(input,tf.float32), shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # todo 第一层卷积
    net = conv(name='conv1',input=input,w=[3, 3, 1, 32], b=32,is_train=is_train)  # 图像(-1,60,160,1) >>(60,160,32)
    net = max_pool(name='pool1',x=net,k=2) # 池化 >>(30,80,32)
    net = tf.nn.dropout(net, rate=1 - keep_prob) # dropout 防止过拟合

    # todo 第二层卷积
    net = conv(name='conv2', input=net,w=[3,3,32,64], b=64,is_train=is_train)  # (30,80,32) >>(30,80,64)
    net = max_pool(name='pool2',x=net,k=2) # 池化 >>[15,40,64]
    net = tf.nn.dropout(net, rate=1 - keep_prob) # dropout 防止过拟合

    # todo 第三层卷积
    net = conv(name='conv3', input=net,w=[3, 3, 64, 64], b=64,is_train=is_train) # (15,40,64) >>(15,40,64)
    net = max_pool(name='pool3', x=net,k=2) # 池化>>[8,20,64]
    net = tf.nn.dropout(net, rate=1 - keep_prob) # dropout 防止过拟合

    # todo Fully connected layer
    net = flatten(net)  # >>[-1, 8 * 20 * 64]
    net = tf.layers.dense(net, units=1024, activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.05), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003)) # >>[-1,1024]
    net = tf.nn.dropout(net, rate=1 - keep_prob)

    with tf.variable_scope('fc21'):
        net1 = tf.layers.dense(net, units=LABEL_DEPTH, activation=None, name='logits',kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.05), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))  # >>[-1,62]

    with tf.variable_scope('fc22'):
        net2 = tf.layers.dense(net, units=LABEL_DEPTH, activation=None, name='logits',kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.05), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))  # >>[-1,62]

    with tf.variable_scope('fc23'):
        net3 = tf.layers.dense(net, units=LABEL_DEPTH, activation=None, name='logits',kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.05), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))  # >>[-1,62]

    with tf.variable_scope('fc24'):
        net4 = tf.layers.dense(net, units=LABEL_DEPTH, activation=None, name='logits',kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.05), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))  # >>[-1,62]

    return net1, net2, net3, net4  # net1:(-1,LABEL_DEPTH=62)

def losses(logit1, logit2, logit3, logit4, labels):
    """
    构建损失函数
    :param logit1:
    :param logit2:
    :param logit3:
    :param logit4:
    :param labels:
    :return:
    """
    labels = tf.convert_to_tensor(labels,tf.int32)
    with tf.variable_scope('loss1'):
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit1, labels=labels[:, 0])
        loss1 = tf.reduce_mean(ce, name='loss1')
    with tf.variable_scope('loss2'):
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit2, labels=labels[:, 1])
        loss2 = tf.reduce_mean(ce, name='loss2')
    with tf.variable_scope('loss3'):
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit3, labels=labels[:, 2])
        loss3 = tf.reduce_mean(ce, name='loss3')
    with tf.variable_scope('loss4'):
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit4, labels=labels[:, 3])
        loss4 = tf.reduce_mean(ce, name='loss4')
    return loss1, loss2, loss3, loss4

def create_optimizer(loss1, loss2, loss3, loss4, learning_rate=0.01):
    """
    构建模型优化器
    :param loss1:
    :param loss2:
    :param loss3:
    :param loss4:
    :param learning_rate:
    :return:
    """
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        with tf.name_scope('optimizer1'):
            optimizer1 = tf.train.AdamOptimizer(learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_opt1 = optimizer1.minimize(loss1, global_step=global_step)
        with tf.name_scope('optimizer2'):
            optimizer2 = tf.train.AdamOptimizer(learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_opt2 = optimizer2.minimize(loss2, global_step=global_step)
        with tf.name_scope('optimizer3'):
            optimizer3 = tf.train.AdamOptimizer(learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_opt3 = optimizer3.minimize(loss3, global_step=global_step)
        with tf.name_scope('optimizer4'):
            optimizer4 = tf.train.AdamOptimizer(learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_opt4 = optimizer4.minimize(loss4, global_step=global_step)

    return train_opt1, train_opt2, train_opt3, train_opt4

def create_accuracy(logit1, logit2, logit3, logit4, labels):
    """
    计算模型准确率
    :param logit1:
    :param logit2:
    :param logit3:
    :param logit4:
    :param labels:
    :return:
    """
    # 对logits进行组合 shape=[4 *batch_size, 62]
    logits_all = tf.concat([logit1, logit2, logit3, logit4], axis=0)

    # fixme 将标签转置 再拉平。 [batch_size, 4] --> [4, batch_size] --> [4*batch_size,]
    labels = tf.convert_to_tensor(labels, tf.int32) #將数组转换为张量
    labels_all = tf.reshape(tf.transpose(labels), shape=[-1])  #矩阵转置,将[batch_size, 7] --> [7, batch_size],,在拉平

    # 计算准确率。
    with tf.variable_scope('accuracy'):
        # 具体有哪一些是最大值。
        correct = tf.nn.in_top_k(predictions=logits_all, targets=labels_all, k=1)
        # 数据类型转换
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy

def train(model_save_path='./models/model',
          log_path='./models/log_dir',
          trainData_dir='./source_data/train',
          validData_dir='./source_data/valid',
          label_num=4,
          label_depth=62,
          image_height=60,
          image_weight=160,
          batch_size=50,
          keep_prob=0.75,
          lr=0.001,
          epoches=1
          ):
    create_dir_path(path=log_path) #创建训练日志保存路径

    # 读取图片数据 train_image:(50, 60, 160, 1), train_label:(50,4)
    train_image, train_label = generateCnn_image_label_batch(image_path=trainData_dir, label_num=label_num, label_depth=label_depth,batch_size=batch_size,image_height=image_height,image_weight=image_weight, is_make_label_onehot=False)
    valid_image, valid_label = generateCnn_image_label_batch(image_path=validData_dir, label_num=label_num, label_depth=label_depth,batch_size=batch_size,image_height=image_height,image_weight=image_weight, is_make_label_onehot=False)

    training = tf.placeholder(tf.bool, shape=None, name='bn_training')  # 训练时需要更新参数,但测试时不需要更新参数
    train_or_valid = tf.placeholder_with_default(False, shape=None, name='is_train') # 判断是传入训练数据  还是 测试数据
    # 基于是否训练操作，做一个选择（选择训练数据集 或者 测试数据集）
    x = tf.cond(train_or_valid, lambda: train_image, lambda: valid_image)
    y = tf.cond(train_or_valid, lambda: train_label, lambda: valid_label)

    # 2、todo 调用cnn()函数构建模型---logit1:(batch_size=50,label_depth=62)
    logit1, logit2, logit3, logit4 = cnn(input=x, keep_prob=keep_prob, LABEL_DEPTH=label_depth,is_train=training)
    # 3,构建损失
    loss1, loss2, loss3, loss4=losses(logit1=logit1, logit2=logit2, logit3=logit3, logit4=logit4, labels=y)
    # fixme 可视化模型损失
    tf.summary.scalar(name='train_loss', tensor=(loss1+loss2+loss3+loss4), collections=['train'])

    # 构建模型优化器
    op1, op2, op3, op4 = create_optimizer(loss1, loss2, loss3, loss4,learning_rate=lr) \
    # 计算准确率
    train_acc= create_accuracy(logit1, logit2, logit3, logit4, labels=y)
    # fixme 可视化模型损失
    tf.summary.scalar(name='train_acc', tensor=train_acc, collections=['train'])

    # 可视化代码
    train_summary = tf.summary.merge_all('train')

    # 构建saver类
    saver = tf.train.Saver(max_to_keep=2)
    create_dir_path(path=model_save_path)
    # 二、执行会话。
    with tf.Session() as sess:
    # 0、断点继续训练(恢复模型)
        ckpt = tf.train.get_checkpoint_state(model_save_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('加载持久化模型，断点继续训练!')
        else:
            # 1、初始化全局变量
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            print('没有持久化模型，从头开始训练!')

        # FileWriter 的构造函数中包含了参数log_dir，申明的所有事件都会写到它所指的目录下
        summary_writer = tf.summary.FileWriter(logdir=log_path, graph=sess.graph)

        # 开启协调器,使用start_queue_runners 启动队列填充
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        step = 1
        # 2、构建迭代的循环
        # ****************************************************************************************************************************************
        try:
            for i in range(epoches):  # 迭代几轮
                for j in range(60):  # 共3000张图片,每个批次50张图片,共60批次
                    print('第{}轮,第{}个批次--------'.format(i + 1, j + 1))
                    _, _, _, _, loss1_, loss2_, loss3_, loss4_, train_acc_, train_summary_ = sess.run(
                    [op1, op2, op3, op4, loss1, loss2, loss3, loss4, train_acc,train_summary], feed_dict={training: True,       # 训练模式,参数可更新
                                                                                             train_or_valid:True  # 训练模式,选择训练数据,而非测试数据
                                                                                             })

                    if step % 5 == 0:
                        all_loss = loss1_ + loss2_+ loss3_+ loss4_
                        summary_writer.add_summary(train_summary_, global_step=step) #每5批就生成一次日志文件
                        print('Step:{} - All Loss:{:.5f} - Train Acc:{:.4f}'.format(step+1, all_loss, train_acc_))

                    if step % 10 == 0: # todo 模型持久化 每训练够10轮就保存一次模型
                        file_name = '_{}_model.ckpt'.format(step)
                        save_file = os.path.join(model_save_path, file_name)
                        saver.save(sess=sess, save_path=save_file, global_step=step)
                        print('model saved to path:{}'.format(save_file))

                    step += 1  # 每训练一批,step就+1,主要是为了tf.summary和模型持久化
                summary_writer.flush() # 每训练一轮,就将日志文件从内存冲洗到磁盘一次
        except tf.errors.OutOfRangeError:
            print("done")
        finally:
            coord.request_stop()
        coord.join(threads)
        # **************************************************************************************************************************************
        summary_writer.close()
        sess.close()
        print('训练运行成功.................................................')


if __name__ == '__main__':
    train()
