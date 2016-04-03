__author__ = 'luoshalin'

import json
import re
from classes import Review
from scipy.sparse import *


def get_review_list(stopword_fpath, input_fpath, class_num, eval_data_frac):
    review_list = []  # list of Review obj
    stars_list = []  # list of stars for all reviews
    # stars_row = []
    # stars_col = []
    # stars_data = []

    f = open(stopword_fpath)
    stopword_lines = f.readlines()
    f.close()
    stopword_set = set(line.strip() for line in stopword_lines)

    line_num = 0  # i-th line
    print 'preprocess - line22'

    with open(input_fpath) as input_f:
        line = input_f.readline()
        while line != '':
            # STEP1 - Extract useful fields
            review = json.loads(line)
            review_id = review['review_id']
            review_text = review['text']
            review_stars = int(review['stars'])

            # STEP2 - Tokenize the text field and remove the stop words (see stopword.list)
            # Turn all the tokens into lower case; Remove punctuation in the tokens; Remove tokens contain numbers
            review_text = review_text.lower()
            review_text = re.sub('[^A-Za-z0-9\s]+', '', review_text)
            review_token_list = review_text.split()
            tmp_review_token_list = []
            for token in review_token_list:
                if token not in stopword_set and not any(i.isdigit() for i in token):
                    tmp_review_token_list.append(token)
            review_token_list = tmp_review_token_list
            new_review = Review(review_id, review_stars, review_token_list)
            # update reviews
            review_list.append(new_review)
            # update stars
            stars_list.append(review_stars)
            # # star matrix
            # stars_col.append(review_stars-1)
            # stars_row.append(line_num)
            # stars_data.append(1)
            line = input_f.readline()
            line_num += 1

    # print 'preprocess - line56'
    # # split the data into train & eval parts
    # train_data_amt = int(round(line_num * (1.0 - eval_data_frac)))
    #
    # train_review_list = review_list[:train_data_amt]
    # eval_review_list = review_list[train_data_amt:]
    #
    # train_stars_list = stars_list[:train_data_amt]
    # eval_stars_list = stars_list[train_data_amt:]
    # print 'preprocess - line65'
    #
    # train_stars_data = stars_data[:train_data_amt]
    # eval_stars_data = stars_data[train_data_amt:]
    # train_stars_row = stars_row[:train_data_amt]
    # tmp_eval_stars_row = stars_row[train_data_amt:]
    # print 'preprocess - line71'
    # eval_stars_row = [elem - train_data_amt for elem in tmp_eval_stars_row]   # ??? right or not?? ..for test_stars_M row size = line_num - train_data_amt
    # print 'preprocess - line73'
    # train_stars_col = stars_col[:train_data_amt]
    # eval_stars_col = stars_col[train_data_amt:]

    # train_stars_M = csr_matrix((train_stars_data, (train_stars_row, train_stars_col)), shape=(train_data_amt, class_num))
    # eval_stars_M = csr_matrix((eval_stars_data, (eval_stars_row, eval_stars_col)), shape=(line_num - train_data_amt, class_num))

    # stars_M = csr_matrix((stars_data, (stars_row, stars_col)), shape=(line_num, class_num))
    # return review_list, stars_list, stars_M

    return review_list, stars_list, line_num


def print_analysis(review_list):
    # STEP3 - Calculate some statistics to verify your implementation.
    token_num_dic = dict()
    stars_list = [0, 0, 0, 0, 0, 0]
    for review in review_list:
        stars = getattr(review, 'stars')
        stars_list[stars] += 1
        tokens = getattr(review, 'tokens')
        for t in tokens:
            if t not in token_num_dic:
                token_num_dic[t] = 1
            else:
                token_num_dic[t] += 1
    tokens = sorted(token_num_dic.items(), key=lambda x: -x[1])

    # 1: top 9 most frequent tokens and corresponding counts
    print "Top 9 most frequent tokens and corresponding counts"
    for i in range(9):
        print tokens[i]

    # 2:number of training instances with 1 star, 2 stars, 3 stars, 4 stars and 5 stars
    for i in range(1, 6):
        print str(i) + " stars review count: " + str(stars_list[i])