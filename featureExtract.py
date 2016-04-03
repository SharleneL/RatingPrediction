__author__ = 'luoshalin'

from scipy.sparse import *


# def get_feature_M(review_list, stars_list, method, feature_num):
#     print 'featureExtract - 7'
#     # feature_set, review_token_freq_dic_list = get_feature_set_and_review_freq(review_list, method, feature_num)
#     get_feature_set_and_review_freq(review_list, stars_list, method, feature_num)
#     print 'featureExtract - 9'
#     # M = get_feature_M_helper(feature_set, review_token_freq_dic_list)
#     print 'featureExtract - 11'
#     # return M


# function to get feature_set and reviews represented by freq dicts
def gen_libsvm_file(review_list, stars_list, method, feature_num):
    feature_str_index_dic = dict()  # dic saves top-2000 most frequent tokens <feature_str, feature_index>
    review_token_freq_dic_list = []  # list of dicts, each dicts is for one review, containing <token, freq>
    feature_dic = dict()  # dict saves all original features <token, corpus_ctf_cnt / corpus_df_cnt>

    print 'featureExtract - 22'
    if method == 'ctf':
        for review in review_list:
            tokens = getattr(review, 'tokens')
            review_token_freq_dic = dict()
            for t in tokens:
                # update feature dic
                if t not in feature_dic:
                    feature_dic[t] = 1
                else:
                    feature_dic[t] += 1
                # update review dic
                if t not in review_token_freq_dic:
                    review_token_freq_dic[t] = 1
                else:
                    review_token_freq_dic[t] += 1
            review_token_freq_dic_list.append(review_token_freq_dic)

    elif method == 'df':
        for review in review_list:
            tokens = getattr(review, 'tokens')
            tokens_set = set(tokens)  # use set here, to judge whether a token appears in one review

            # update feature dic
            for t in tokens_set:
                if t not in feature_dic:
                    feature_dic[t] = 1
                else:
                    feature_dic[t] += 1

            # update review dic
            review_token_freq_dic = dict()
            for t in tokens:
                if t not in review_token_freq_dic:
                    review_token_freq_dic[t] = 1
                else:
                    review_token_freq_dic[t] += 1
            review_token_freq_dic_list.append(review_token_freq_dic)

    else:
        print 'ERROR: Invalid feature extraction method: ' + method

    print 'featureExtract - 64'

    sorted_feature_list = sorted(feature_dic.items(), key=lambda x: -x[1])

    print 'featureExtract - 68'

    for i in range(feature_num):  # 10 for testing, or 2000 for experiment
        feature_str_index_dic[sorted_feature_list[i][0]] = i

    # output to a dict file
    with open(method+'_top_features.list', 'w') as feature_f:
        for i in range(feature_num):  # 10 for testing, or 2000 for experiment
            feature_f.write(str(sorted_feature_list[i][0]) + '\t' + str(sorted_feature_list[i][1]) + '\n')

    print 'featureExtract - 73'

    # get feature_str_index_dic & review_token_freq_dic_list - next: output useful features to file

    output_fpath = method + '_libsvm'+'.out'
    with open(output_fpath, 'w') as output_f:
        for i in range(len(review_token_freq_dic_list)):  # each elem d in the list is a dict for one review
            output_str = ''
            d = review_token_freq_dic_list[i]

            for token, freq in d.iteritems():
                if token in feature_str_index_dic:
                    output_str = output_str + str(feature_str_index_dic[token]) + ':' + str(d[token]) + ' '
            output_str = str(stars_list[i]) + ' ' + output_str.strip() + '\n'
            output_f.write(output_str)

    print 'featureExtract - 89'

    return output_fpath

    # return feature_set, review_token_freq_dic_list


# get the feature matrix: row - one review, col - token's index in feature dic, value - occurence frequency
def get_feature_M_helper(feature_set, review_token_freq_dic_list):
    row = []
    col = []
    data = []
    for i in range(len(review_token_freq_dic_list)):  # each elem d in the list is a dict for one review
        d = review_token_freq_dic_list[i]
        feature_list = list(feature_set)

        for token, freq in d.iteritems():
            if token in feature_set:
                row.append(i)
                col.append(feature_list.index(token))
                data.append(d[token])
    M = csr_matrix((data, (row, col)), shape=(len(review_token_freq_dic_list), len(feature_set)))
    return M
