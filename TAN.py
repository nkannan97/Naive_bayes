from naive_bayes import *
import numpy as np
from scipy.io import arff
import pandas as pd
import math
from MST import *


def get_p_xi_xj(fi_mat, fj_mat, fi_range, fj_range):
    df = pd.DataFrame(index=list(fi_range), columns=list(fj_range))
    df = df.fillna(0)
    for feature_i in range(len(fi_range)):
        for feature_j in range(len(fj_range)):
            for data_instance in range(len(fi_mat)):
                if fi_mat[data_instance] == fi_range[feature_i] and fj_mat[data_instance] == fj_range[feature_j]:
                    df[fj_range[feature_j]].loc[fi_range[feature_i]] += 1

    columns = [col for col in df if df[col].dtype.kind != 'O']
    df[columns] += 1
    prob_df = df.divide(df.values.sum())
    # print(prob_df)
    return prob_df


def get_p_xxy(xi_range, xj_range, feature_range, feature_matrix, label_matrix, f1_index, f2_index):
    p_xxy_dict = {}
    for i in xi_range:
        for j in xj_range:
            for y in feature_range:
                p_xxy_dict[(i, j, y)] = 0

    for value in range(len(feature_matrix)):
        p_xxy_dict[(feature_matrix[value][f1_index], feature_matrix[value][f2_index], label_matrix[value])] += 1

    for key in p_xxy_dict.keys():
        p_xxy_dict[key] += 1

    sum_count = sum(p_xxy_dict.values())
    for in_dict in p_xxy_dict.keys():
        p_xxy_dict[in_dict] = float((p_xxy_dict[in_dict]) / sum_count)

    return p_xxy_dict


def convert_to_dataframe(p):
    key_lst = []
    temp_lst = []
    for key in p.keys():
        key_lst.append(key)

    for index in range(len(p[key_lst[0]])):
        df1 = pd.DataFrame.from_dict(p[key_lst[0]][index], orient='index')
        df2 = pd.DataFrame.from_dict(p[key_lst[1]][index], orient='index')
        merged_df = pd.merge(df1, df2, left_index=True, right_index=True)
        merged_df.columns = [key_lst[0], key_lst[1]]
        temp_lst.append(merged_df)
    return temp_lst


def get_p_given_xx_y(features, labels, num_feature_range, p_y, df_pxGy, meta, I):
    p_xixjG_y = {}
    mat_dict = {}
    label = []

    for l_range in range(len(num_feature_range[-1])):
        feature_matrix_subset, label_matrix_subset = construct_subsets(features, labels, features.shape[1],
                                                                       num_feature_range[-1][l_range])
        mat_dict[num_feature_range[-1][l_range]] = feature_matrix_subset

        label.append(num_feature_range[-1][l_range])

    for i in range(len(num_feature_range) - 1):
        # print('---------Iteration-----------------: ', i)
        for j in range(len(num_feature_range) - 1):
            lst = []
            info_gain = 0
            if i != j:
                p_xixjG_y[label[0]] = get_p_xi_xj(mat_dict[label[0]][:, i], mat_dict[label[0]][:, j],
                                                  num_feature_range[i], num_feature_range[j])
                p_xixjG_y[label[1]] = get_p_xi_xj(mat_dict[label[1]][:, i], mat_dict[label[1]][:, j],
                                                  num_feature_range[i], num_feature_range[j])
                p_xx_y = get_p_xxy(num_feature_range[i], num_feature_range[j], num_feature_range[-1], features, labels,
                                   i, j)

                pxGy = df_pxGy[i]
                next_pxGy = df_pxGy[j]
                for x_i in num_feature_range[i]:
                    for x_j in num_feature_range[j]:
                        for y in label:
                            info_gain += (p_xx_y[(x_i, x_j, y)]) * math.log(
                                ((p_xixjG_y[y][x_j].loc[x_i]) / (pxGy[y].loc[x_i] * next_pxGy[y].loc[x_j])), 2)

                I.loc[meta.names()[i], meta.names()[j]] = info_gain
            p_xixjG_y.clear()

    return I


def print_bayes_net_graph(parents, metadata):
    for i in range(len(parents)):
        if parents[i] == None:
            print('{} {}'.format(metadata.names()[i], metadata.names()[-1]))
        else:
            print('{} {} {}'.format(metadata.names()[i], metadata.names()[parents[i]], metadata.names()[-1]))


def compute_edge_weights(feature_matrix, label_matrix, feature_range, metadata, I):
    p_xGy = get_probab_x_given_y(feature_matrix, label_matrix, feature_range)
    df_pxGy = convert_to_dataframe(p_xGy)
    p_y = getprobDistribution(label_matrix, feature_range[-1])
    edge_weight_matrix = get_p_given_xx_y(feature_matrix, label_matrix, feature_range, p_y, df_pxGy, metadata, I)
    edge_weight_matrix[edge_weight_matrix == 0] = -1
    return edge_weight_matrix


def construct_conditional_subsets(f_mat, l_mat, f, y, f_index):
    split = np.logical_and(f_mat[:, f_index] == f, l_mat == y)
    f_subset = f_mat[split, :]
    l_subset = l_mat[split]
    return f_subset, l_subset


def get_condprob(feature_mat, xi_range, y_val, parent_val):
    p_dict = {}
    for xi in xi_range:
        p_dict[(xi, parent_val, y_val)] = 0
        for value in feature_mat:
            if value == xi:
                p_dict[(xi, parent_val, y_val)] += 1

    for key in p_dict.keys():
        p_dict[key] += 1

    t_sum = sum(p_dict.values())
    for key in p_dict.keys():
        p_dict[key] = 1.0 * p_dict[key] / t_sum

    # print(p_dict)
    return p_dict


def get_p_xiG_xj_y(features, labels, f_index, parent_index, f_range):
    p_xiG_xj_y = []
    cpt_results = {}
    # for i_dict in range(len(f_range)):
    #     p_xiG_xj_y[i_dict] = []

    for y in range(len(f_range[-1])):
        for p_index in range(len(f_range[parent_index])):
            feature_sub, label_sub = construct_conditional_subsets(features, labels, f_range[parent_index][p_index],
                                                                   f_range[-1][y], parent_index)
            p_xiG_xj_y.append(
                get_condprob(feature_sub[:, f_index], f_range[f_index], f_range[-1][y], f_range[parent_index][p_index]))

    for d in p_xiG_xj_y:
        cpt_results.update(d)

    # print(cpt_results)
    return cpt_results


def compute_conditional_probability(feature_matrix, label_matrix, feature_range, parents):
    # iterate through all the features
    p = {}
    for index in range(len(feature_range) - 1):
        if parents[index] == None:
            p_xgy = get_probab_x_given_y(feature_matrix, label_matrix, feature_range)
            df_pxgy = convert_to_dataframe(p_xgy)
            p[index] = df_pxgy[index]
        else:
            p[index] = get_p_xiG_xj_y(feature_matrix, label_matrix, index, parents[index], feature_range)

    p[index + 1] = getprobDistribution(label_matrix, feature_range[-1])
    # print(check)
    return p


def compute_prob(test_mat, p_dist, parents, y):
    p = p_dist[len(p_dist) - 1][y]

    for test_instance in range(len(test_mat)):
        p_xi = test_mat[test_instance]
        if parents[test_instance] == None:
            val = p_dist[test_instance][y].loc[p_xi]
        else:
            p_xj = test_mat[parents[test_instance]]
            val = p_dist[test_instance][(p_xi, p_xj, y)]

        p *= val

    return p


def predict_bayes_net(test_features, cond_prob, feature_range, parents):
    class_label = np.zeros((test_features.shape[0], 1), dtype="<U20")
    probability_values = np.zeros((test_features.shape[0], 1))
    prediction = np.zeros((len(feature_range[-1]), 1))

    for test_size in range(test_features.shape[0]):
        for y in range(len(feature_range[-1])):
            prediction[y] = compute_prob(test_features[test_size, :], cond_prob, parents, feature_range[-1][y])

        prediction = np.divide(prediction, np.sum(prediction))
        max_prediction = np.argmax(prediction)
        probability_values[test_size] = prediction[max_prediction]
        class_label[test_size] = feature_range[-1][max_prediction]

    return class_label, probability_values


def print_TAN_results(predicted_class_label, prob_values, actual_label, meta):
    print()
    label_range = meta[meta.names()[-1]][1]
    correct_count = 0
    for num_instances in range(len(actual_label)):
        prediction = predicted_class_label[num_instances]
        truth = actual_label[num_instances]
        print('{} {} {:0.12f}'.format(prediction[0].strip('"\''), truth.strip('"\''), prob_values[num_instances][0]))

    for length in range(len(actual_label)):
        if actual_label[length] == predicted_class_label[length]:
            correct_count += 1
        else:
            continue
    print()
    print('{}'.format(correct_count))
