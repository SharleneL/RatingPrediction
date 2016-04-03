# the dis(W_new, W_old, ..) method needs to be implemented

__author__ = 'luoshalin'

import numpy as np
import math


# x_M is the matrix of features <n * feature_num:2000>
# y_M is a matrix of stars (each row corresponding to one data point, each col represents one class:5)
# W is the parameter matrix <feature_num:2000, class_num:5>
def lr_train_param(x_M, y_M, W, eval_x_M, eval_y_M, lmd, alpha, threshold, method, batch_size):
    n = x_M.get_shape()[0]    # total number of data points
    c = W.shape[1]            # total number of classes

    # ====== / testing start / ====== #
    # new_W = np.zeros(shape=(W.shape[0], c))
    # for i in range(n):  # for each data point
    #     x_i = x_M.getrow(i).toarray()  # get the feature on i-th row
    #     y_i = y_M.getrow(i).toarray()  # get the class on i-th row
    #     denom = sum(np.exp(np.dot(W.T, x_i.T)))
    #     # ::: combined ::: #
    #     new_W += np.dot(x_i.T, (y_i.T - (np.exp(np.dot(W.T, x_i.T))) / denom).T)
    #     # ::: seperate ::: #
    #     # a = np.dot(W.T, x_i.T)
    #     # b = np.exp(a)
    #     # c = b / denom
    #     # d = y_i.T - c
    #     # e = np.dot(x_i.T, d.T)
    #
    # new_W -= lmd * W
    # print new_W
    # ====== / testing end / ====== #

    W_new = np.zeros(shape=(W.shape[0], c))
    if method == 'sga':
        for i in range(n):  # for each data point
            x_i = x_M.getrow(i).toarray()  # get the feature on i-th row: <1 * #feature>
            y_i = y_M.getrow(i).toarray()  # get the class on i-th row: <1 * #class>
            denom = sum(np.exp(np.dot(W.T, x_i.T)))  # sum( <#class * #feature> * <#feature * 1> )
            numer = np.exp(np.dot(W.T, x_i.T))
            p = numer / denom
            W_g = np.dot(x_i.T, (y_i.T - p).T) - lmd * W

            W_new = W + alpha * W_g

            # judge with stopping criteria
            ll_old = cal_ll(eval_x_M, eval_y_M, W, lmd)
            ll_new = cal_ll(eval_x_M, eval_y_M, W_new, lmd)
            diff = abs(ll_old - ll_new)
            print 'Round #' + str(i) + ' ll_old:' + str(ll_old) + ' ll_new:' + str(ll_new) + ' DIFF:' + str(diff)
            if diff < threshold:
                break
            else:
                W = W_new
            # print ':::::: ITERATION ' + str(i) + ' ::::::'
            # print W_new

    elif method == 'bsga':
        iter = 0
        while batch_size * (iter + 1) < n:
            W_g = np.zeros(shape=(W.shape[0], c))
            for i in range(batch_size * iter, batch_size * (iter+1)):
                x_i = x_M.getrow(i).toarray()  # get the feature on i-th row
                y_i = y_M.getrow(i).toarray()  # get the class on i-th row
                denom = sum(np.exp(np.dot(W.T, x_i.T)))
                W_g += np.dot(x_i.T, (y_i.T - (np.exp(np.dot(W.T, x_i.T))) / denom).T)
            W_g -= lmd * W
            iter += 1

            W_new = W + alpha * W_g

            # judge with stopping criteria
            ll_old = cal_ll(eval_x_M, eval_y_M, W, lmd)
            ll_new = cal_ll(eval_x_M, eval_y_M, W_new, lmd)
            diff = abs(ll_old - ll_new)
            print 'Round #' + str(i) + ' ll_old:' + str(ll_old) + ' ll_new:' + str(ll_new) + ' DIFF:' + str(diff)
            if diff < threshold:
                break
            else:
                W = W_new
            # print ':::::: ITERATION ' + str(i) + ' ::::::'
            # print W_new

    else:
        print 'ERROR: Invalid gradient ascent method: ' + method

    return W_new


def dis(W_old, W_new, W_g):
    # ????
    # return sum(sum(W_g))
    return 0.5


# function to predict testing data's class
# x_M: data matrix; W: parameter matrix
def lr_predict(x_M, W, method):
    n = x_M.get_shape()[0]    # total number of data points
    c = W.shape[1]            # total number of classes

    numer_M = np.exp(np.dot(W.T, x_M.T.toarray()))  # x_M is sparse, so need to convert to np array
    denom_arr = np.sum(numer_M, axis=0)  # sum by column
    p_M = numer_M / denom_arr  # p_M: each row is one class, each col is one data point, data is the prob of this data classified into this class

    if method == 'hard':
        pred_res = p_M.argmax(axis=0) + 1  # find the index of max value along each col, then + 1
        return pred_res

    elif method == 'soft':
        class_arr = np.asarray(list(range(1, c+1)))  # an arr [1, 5]
        pred_res = np.dot(class_arr, p_M)  # a [1*n] arr, get weighed sum for each data point
        return pred_res

    else:
        print 'ERROR: Invalid predict method: ' + method


# get the ll of param M
def cal_ll(eval_x_M, eval_y_M,  W, lmd):
    n = eval_x_M.get_shape()[0]    # total number of evaluation data points
    c = W.shape[1]            # total number of classes

    lp_sum = 0.0
    for i in range(n):  # for each data point
        x_i = eval_x_M.getrow(i).toarray()  # get the feature on i-th row: <1 * #feature>
        y_i = eval_y_M.getrow(i).toarray()  # get the class on i-th row: <1 * #class>
        w_c = np.dot(y_i, W.T)  # get the corresponding param line of current data's label: <1 * #feature>
        denom = sum(np.exp(np.dot(W.T, x_i.T)))  # sum( <#class * #feature> * <#feature * 1> )
        numer = math.exp(np.dot(w_c, x_i.T)[0][0])
        p = numer / denom
        lp_sum += math.log(p)  # e-base log

    w_len_sum = np.sum(np.square(W))
    ll = lp_sum - float(lmd * w_len_sum) / 2
    return ll