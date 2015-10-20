#! coding=utf-8
"""
	CNN(卷积神经网络）的python实现
	本代码是通过对mnist手写字体进行识别的一个应用
	@date 2015-10-18
	@author Pony
"""

# 导入必要的包
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

# 导入之前的逻辑回归代码、多层感知器代码
from logis import LogisticRegression, load_data
from MLP import HiddenLayer

# 定义卷积网络类
class LeNetConvPoolLayer(object):
    """卷积网络的Pool层 """

    # 构造函数
    #	input:输入(4D)
    #   filter_shape:filter维度(4D):number of filters,number of input feature maps,filter height,filter width
    #   image_shape:图片维度:batch size,number of input feature maps,image height,image width
    #   poolsize:pool矩形大小
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
	
	assert image_shape[1]=filter_shape[1]   #断言
        self.input = input   

        # 对于每一个隐藏层神经元，有"num input feature maps * filter height * filter width"个输入
        fan_in = numpy.prod(filter_shape[1:])
        
	# 有"num output feature maps * filter height * filter width" /pooling size个输出
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /numpy.prod(poolsize))
        
	# 随机初始化w参数，w维度为filter_shape一样
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # 初始化偏置参数b，长度为filter_map的个数
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # 对input和filter进行卷积
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # 对每个特征层进行抽样降低维度，使用max pool方法
            pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # 首先将偏置b变形为维度（1，n_filters,1,1)的矩阵，然后进行相加
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # 将W b保存到params数组中
        self.params = [self.W, self.b]

        self.input = input


def evaluate_lenet5(learning_rate=0.1, n_epochs=200,
                    dataset='mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=500):
    """ 在mnist数据集上验证模型

    :type learning_rate: float
    :param learning_rate: 用于梯度下降的学习率大小（或步长）

    :type n_epochs: int
    :param n_epochs: 最大的优化周期

    :type dataset: string
    :param dataset: 数据集名称

    :type nkerns: list of ints
    :param nkerns: 每一层的核的数目
    """

    # 随机化种子
    rng = numpy.random.RandomState(23455)

    # 导入数据
    datasets = load_data(dataset)

    # mnist数据集有三种，分别是train、valid、test数据集，可以分别导入它们
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # 每种类型数据集的数目，并计算它们的batch数目
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # 分配变量，用于计数batch
    index = T.lscalar()  
    
    #x:图像输入,y:标签输出
    x = T.matrix('x')   
    y = T.ivector('y')  

    ######################
    # 下面开始构建真正的模型#
    ######################
    print '... building the model'

    # 将输入数据变形为四维矩阵(batch_size,1,28,28)，其中28*28为图片的大小
    layer0_input = x.reshape((batch_size, 1, 28, 28))
    
    # 构建第一个卷积-pooling层
    # filtering使得图片的大小减小为 (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling使得它变为 (24/2, 24/2) = (12, 12)
    # 输出的是4D tensor,形状是 (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    # 构建第二个卷积-pooling层
    # filtering使得图片的大小减小为 (12-5+1, 12-5+1) = (8, 8)
    # maxpooling使得它变为 (8/2, 8/2) = (4, 4)
    # 输出的是4D tensor，形状是 (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # 隐藏层是全连接的，它的输入是形状为 (batch_size, num_pixels) 
    layer2_input = layer1.output.flatten(2)

    # 构建一个全连接的隐藏层，激活函数为theano.tensor.tanh
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=500,
        activation=T.tanh
    )

    # 对全连接层进行分类
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

    # 损耗函数
    cost = layer3.negative_log_likelihood(y)

    # 计算模型训练时并产生的误差 
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # 验证模型
    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # 合并参数
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # 损耗函数对参数求导
    grads = T.grad(cost, params)

    # 使用SGD算法来更新参数 
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    # 训练模型
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # 训练模型 #
    ###############
    print '... training'
    # 提前停止训练的参数
    patience = 10000  
    patience_increase = 2 
                          
    improvement_threshold = 0.995  
                                   
    validation_frequency = min(n_train_batches, patience / 2)
                                  
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # 计算validation损耗
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # 如果已经得到最好的validation
               if this_validation_loss < best_validation_loss:

                    # 提高patience
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # 设置最好的validation_loss
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # 测试
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
