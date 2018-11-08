import numpy as np
from scipy.io import arff
import pandas as pd
import sys

'''
Reading training and testing data
'''


def read_data(file_name):
    data, metadata = arff.loadarff(file_name)

    features = []
    labels = []
    number_val_for_feature = []
    label_range = metadata[metadata.names()[-1]][1]
    for instances in range(len(data)):
        labels.append(str(data[instances][-1], 'utf-8'))

        f = []
        for feature in range(len(metadata.names()) - 1):
            # f.append(metadata[metadata.names()[feature]][1].index(str(data[instances][feature], 'utf-8')))
            f.append(str(data[instances][feature], 'utf-8'))
        features.append(f)
    for value in range(len(metadata.names())):
        number_val_for_feature.append(metadata[metadata.names()[value]][1])

    label_matrix = np.array(labels)
    feature_matrix = np.array(features)
    return feature_matrix, label_matrix, number_val_for_feature, metadata


'''
calculating P(Y) and P(X|Y) using dictionary
'''


def getprobDistribution(matrix, feature_range_val):
    i = 0
    p = {}
    count_prob = {}
    probab_dist = []
    for value in matrix:
        if value not in p:
            p[value] = 1
        else:
            p[value] += 1

    for i in range(len(feature_range_val)):
        if feature_range_val[i] not in p.keys():
            p[feature_range_val[i]] = 1
        else:
            p[feature_range_val[i]] += 1
    # print(sum(p.values()))
    for counts in p.keys():
        # probab_dist.append(1.0*p[counts]/len(matrix))
        count_prob[counts] = float((p[counts]) / (sum(p.values())))
    # prob_dist_array = np.array(probab_dist)
    return count_prob


'''
Dividing data set into 2 separate columns depending on the class label
'''


def construct_subsets(feature_matrix, label_matrix, idx, val):
    index_lst = []
    if idx == feature_matrix.shape[1]:
        # print(label_matrix)
        # print(val)
        split = (label_matrix == val)

    feature_matrix_subset = feature_matrix[split, :]
    label_matrix_subset = label_matrix[split]
    return feature_matrix_subset, label_matrix_subset


'''
Helper method for P(X|Y)
'''


def get_probab_x_given_y(feature_matrix, label_matrix, feature_range_vals):
    probab_x_given_y = {}
    count_dict = {}
    for value in range(len(feature_range_vals[-1])):
        temp_list = []
        feature_matrix_subset, label_matrix_subset = construct_subsets(feature_matrix, label_matrix,
                                                                       feature_matrix.shape[1],
                                                                       feature_range_vals[-1][value])

        for num_feature in range(len(feature_range_vals) - 1):
            count = getprobDistribution(feature_matrix_subset[:, num_feature], feature_range_vals[num_feature])

            # print(conditional_prob)
            temp_list.append(count)

        probab_x_given_y[feature_range_vals[-1][value]] = temp_list

    # print(len(probab_x_given_y['metastases']))
    return probab_x_given_y


'''
predictions
'''


def predictor(x_test, prob_y, prob_xGy, label_range):
    test_prob = np.zeros((len(label_range), 1))
    for y_val in range(len(label_range)):
        test_prob[y_val] = prob_y[label_range[y_val]]
        for num in range(len(x_test)):
            prob_temp = prob_xGy[label_range[y_val]][num]
            test_prob[y_val] *= prob_temp[x_test[num]]

    prediction = np.divide(test_prob, np.sum(test_prob))
    max_prediction = np.argmax(prediction)
    predict_prob = prediction[max_prediction]
    return label_range[max_prediction], predict_prob


def predict(test_data, probab_Y, prob_xGy, meta):
    class_label = np.zeros((test_data.shape[0], 1), dtype="<U20")
    probability_values = np.zeros((test_data.shape[0], 1))
    for test_instance in range(len(test_data)):
        class_label[test_instance], probability_values[test_instance] = predictor(test_data[test_instance, :], probab_Y,
                                                                                  prob_xGy, meta[meta.names()[-1]][1])
    # print(class_label)
    # print(probability_values)
    return class_label, probability_values


'''
printing the test results
'''


def print_naiveBayes_graph(metadata):
    for i in range(len(metadata.names()) - 1):
        print('{} {}'.format(metadata.names()[i], metadata.names()[-1]))

    print()


def print_naiveBayes_results(predicted_class_label, prob_values, actual_label, meta):
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
