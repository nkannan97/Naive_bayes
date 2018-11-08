import numpy as np
from naive_bayes import *
from TAN import *
from MST import *


class cross_val():

    def __init__(self, training_data, num_folds):

        self.data = training_data
        self.num_folds = num_folds
        self.folds = {}
        self.indexfolds = {}

    def split_data(self, data, label_mat, label_range):

        label_0 = []
        label_1 = []
        for i in range(len(data)):

            if label_mat[i] == label_range[0]:
                label_0.append(i)
            else:
                label_1.append(i)

        for indices in range(self.num_folds):
            self.indexfolds[indices] = []
            self.folds[indices] = []

        lst = [label_0, label_1]

        for index in range(len(label_range)):
            np.random.shuffle(lst[index])

        ratio_0 = len(label_0) / len(data)
        ratio_1 = 1 - ratio_0

        instances_per_fold = np.round(len(data) / self.num_folds)

        label_0_instances = np.round(ratio_0 * instances_per_fold)
        label_1_instances = instances_per_fold - label_0_instances
        count_0 = 0
        count_1 = 0

        common_lst = []

        for j in range(self.num_folds):
            while count_0 < label_0_instances and label_0 != []:
                common_lst.append(label_0.pop())
                count_0 += 1
            while count_1 < label_1_instances and label_1 != []:
                common_lst.append(label_1.pop())
                count_1 += 1
            count_0 = 0
            count_1 = 0

            self.folds[j] = common_lst
            self.indexfolds[j] = self.folds[j]
            common_lst = []

        if label_0 != [] or label_1 != []:
            self.folds[self.num_folds] = list(set(label_0) | set(label_1))

    def split_train_test(self, fold_index, label_matrix, label_range):

        get_fold_indices = list(self.folds[fold_index])
        get_fold_indices.sort()
        training_data = []
        testing_data = []
        label_test = []
        label_train = []

        for indices in range(len(self.data)):

            if indices in get_fold_indices:
                testing_data.append(self.data[indices])
                label_test.append(label_matrix[indices])
            else:
                training_data.append(self.data[indices])
                label_train.append(label_matrix[indices])

        c = list(zip(training_data, label_train))
        np.random.shuffle(c)
        shuffled_training_data, shuffled_label_data = zip(*c)
        # np.random.shuffle(training_data)
        # print('indices: ', get_fold_indices)
        # print('testing_data: ', testing_data[0])
        return shuffled_training_data, testing_data, shuffled_label_data, label_test


feature_matrix, label_matrix, number_val_for_feature, metadata = read_data('chess-KingRookVKingPawn.arff')
cross_validation = cross_val(feature_matrix, 10)
cross_validation.split_data(feature_matrix, label_matrix, number_val_for_feature[-1])
print(cross_validation.folds)

Nb_accuracy = np.zeros((10, 1))
TAN_accuracy = np.zeros((10, 1))

for index in range(10):
    Nb_count = 0
    TAN_count = 0
    training_data, testing_data, label_train, label_test = cross_validation.split_train_test(index, label_matrix,
                                                                                             number_val_for_feature[-1])
    X_train = np.array(training_data)
    y_train = np.array(label_train)
    X_test = np.array(testing_data)
    y_test = np.array(label_test)

    prob_dict = getprobDistribution(y_train, number_val_for_feature[-1])
    probab_x_given_y = get_probab_x_given_y(X_train, y_train, number_val_for_feature)

    predicted_class_label, prob_values = predict(X_test, prob_dict, probab_x_given_y, metadata)

    I_df = pd.DataFrame(index=metadata.names()[:-1], columns=metadata.names()[:-1])
    I_df = I_df.fillna(0)
    edge_weight_matrix = compute_edge_weights(X_train, y_train, number_val_for_feature, metadata, I_df)
    adjacency_matrix = np.array(edge_weight_matrix)
    MST_parents = prim(adjacency_matrix)
    print_bayes_net_graph(MST_parents, metadata)
    conditional_probability = compute_conditional_probability(X_train, y_train,
                                                              number_val_for_feature, MST_parents)
    class_labels, p_values = predict_bayes_net(X_test, conditional_probability, number_val_for_feature,
                                               MST_parents)
    for i in range(len(y_test)):
        if predicted_class_label[i] == y_test[i]:
            Nb_count += 1
        if class_labels[i] == y_test[i]:
            TAN_count += 1

    Nb_accuracy[index] = 1.0 * Nb_count / len(y_test)
    TAN_accuracy[index] = 1.0 * TAN_count / len(y_test)

print(Nb_accuracy)
print(TAN_accuracy)
