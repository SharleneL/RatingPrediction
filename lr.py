__author__ = 'luoshalin'
# FILE DESCRIPTION: functions for logistic regression training

import numpy as np


# function to get the trained LR parameter matrix
# PARAMS:
# x_M is the matrix of features <n * feature_num:2000>
# y_M is a matrix of stars (each row corresponding to one data point, each col represents one class:5)
# W is the parameter matrix <feature_num:2000, class_num:5>
def lr_train_param(x_M, y_M, W, eval_x_M, eval_y_M, lmd, alpha, threshold, method, batch_size):
    n = x_M.get_shape()[0]    # total number of training data points
    c = W.shape[1]            # total number of classes

    W_new = np.zeros(shape=(W.shape[0], c))
    if method == 'sga':
        for i in range(n):                  # for each data point
            x_i = x_M.getrow(i)             # get the feature on i-th row: <1 * #feature>
            y_i = y_M.getrow(i).toarray()   # get the class on i-th row: <1 * #class>
            numer = np.exp(W.T * x_i.T)
            denom = sum(numer)              # sum( <#class * #feature> * <#feature * 1> )
            W_g = x_i.T * (y_i.T - numer / denom).T - lmd * W

            W_new = W + alpha * W_g
            # judge with stopping criteria
            ll_old = cal_ll(eval_x_M, eval_y_M, W, lmd)
            ll_new = cal_ll(eval_x_M, eval_y_M, W_new, lmd)
            diff = abs((ll_old - ll_new) / ll_old)
            print 'Round #' + str(i) + ' ll_old:' + str(ll_old) + ' ll_new:' + str(ll_new) + ' DIFF:' + str(diff)
            if diff < threshold:
                break
            else:
                W = W_new

    elif method == 'bsga':
        iter = 0
        while batch_size * (iter + 1) < n:
            x_M_batch = x_M[batch_size * iter: batch_size * (iter+1)]               # [L, R) <#batchsize, #feature>
            y_M_batch = y_M[batch_size * iter: batch_size * (iter+1)].toarray()     # [L, R) <#batchsize, #class>
            numer = np.exp(x_M_batch * W)                                           # W: <#feature, #class> => numer: <#batchsize, #class>
            denom = np.sum(numer, axis=1)                                           # <#batchsize, 1>
            sigma_M = x_M_batch.T * (y_M_batch - numer / denom[:, None])            # <#feature, #batchsize> * <#batchsize, #class> => <#feature, #class>
            W_g = sigma_M - lmd * W
            iter += 1

            W_new = W + alpha * W_g
            # judge with stopping criteria
            ll_old = cal_ll(eval_x_M, eval_y_M, W, lmd)
            ll_new = cal_ll(eval_x_M, eval_y_M, W_new, lmd)
            diff = abs((ll_old - ll_new) / ll_old)
            print 'Round #' + str(iter) + ' ll_old:' + str(ll_old) + ' ll_new:' + str(ll_new) + ' DIFF:' + str(diff)
            if diff < threshold:
                break
            else:
                W = W_new

    else:
        print 'ERROR: Invalid gradient ascent method: ' + method

    return W_new


# function to predict testing data's class
# PARAMS:
# x_M: data matrix; W: parameter matrix
def lr_predict(x_M, W):
    n = x_M.get_shape()[0]    # total number of data points
    c = W.shape[1]            # total number of classes

    numer_M = np.exp(W.T * x_M.T)
    denom_arr = np.sum(numer_M, axis=0)             # sum by column
    p_M = numer_M / denom_arr                       # p_M: each row is one class, each col is one data point, data is the prob of this data classified into this class

    pred_res_hard = p_M.argmax(axis=0) + 1          # find the index of max value along each col, then + 1

    class_arr = np.asarray(list(range(1, c+1)))     # an arr [1, 5]
    pred_res_soft = np.dot(class_arr, p_M)          # a [1*n] arr, get weighed sum for each data point

    return pred_res_hard, pred_res_soft


# function to get the log-likelihood of param M
def cal_ll(eval_x_M, eval_y_M,  W, lmd):
    numer = np.exp(eval_x_M * W)                    # W: <#feature, #class> => numer: <#batchsize, #class>
    denom = np.sum(numer, axis=1)                   # <#batchsize, 1>
    p_M_1 = numer / denom[:, None]                  # <#class, #batchsize>
    p_M_2 = np.multiply(eval_y_M.toarray(), p_M_1)
    p_sum = np.sum(np.log(np.sum(p_M_2, axis=1)))
    w_len_sum = np.sum(np.square(W))
    ll = p_sum - float(lmd * w_len_sum) / 2
    return ll