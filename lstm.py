import tensorflow as tf
import numpy as np
import matplotlib
import os
import matplotlib.pyplot as plt
import pandas as pd

tf.set_random_seed(777)  # reproducibility
def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

#timesteps=seq_length=7
#data_dim=16
#output_dim=1
#hidden_dim = 10
#learning_rate = 0.01
#iterations = 500
profit=[]
macket=[]
for j in range(1,113):#120개 기업 
    timesteps=seq_length=5
    data_dim=19
    output_dim=1
    hidden_dim = 10
    learning_rate = 0.01
    iterations = 500

    cv=' .csv'
    xy=np.loadtxt(str(j)+cv, delimiter=',')
    xy=MinMaxScaler(xy)
    x=xy
    y=xy[:,[-1]]
    #test=y[-1]-y[-2]
    #if test> 0:
    #    macket.append(test,j)
    
    

    dataX=[]
    dataY=[]
    for i in range(0,len(y)-seq_length):
        _x=x[i:i+seq_length]
        _y=y[i+seq_length]
        print(_x,'->',_y)
        print(j)
        dataX.append(_x)
        dataY.append(_y)
    train_size=int(len(dataY)*0.68)
    test_size=len(dataY)-train_size

    trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
    trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

    X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
    Y = tf.placeholder(tf.float32, [None, 1])

    cell=tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim,state_is_tuple=True)
    outputs,_states=tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
    Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

    loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())




    for i in range(1000):
        _,l=sess.run([train,loss],feed_dict={X:trainX,Y:trainY})
        print(i,l)
   
    testPredict=sess.run(Y_pred,feed_dict={X:testX})#예측값

    #print(testPredict[-1]-testPredict[-2])


    profit.append([(testPredict[-1]),j])
    dataX.clear()
    dataY.clear()
    #tf.clear_all_variables()
    tf.reset_default_graph()

#print(profit)
profit.sort(reverse=True)
print(profit[0:6]) #상위 5종목 추출






                           
