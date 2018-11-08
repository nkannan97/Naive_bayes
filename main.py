from naive_bayes import *
from TAN import *
from MST import *
import sys


def read_input_args():
    train = str(sys.argv[1])
    test = str(sys.argv[2])
    m = str(sys.argv[3])

    return train, test, m


train_file, test_file, type = read_input_args()
feature_matrix, label_matrix, number_val_for_feature, metadata = read_data(train_file)

if type.strip('"\'') == 'n':
    prob_dict = getprobDistribution(label_matrix, number_val_for_feature[-1])
    probab_x_given_y = get_probab_x_given_y(feature_matrix, label_matrix, number_val_for_feature)
    feature_test_matrix, label_test_matrix, _, test_metadata = read_data(test_file)
    print_naiveBayes_graph(metadata)
    predicted_class_label, prob_values = predict(feature_test_matrix, prob_dict, probab_x_given_y, test_metadata)
    print_naiveBayes_results(predicted_class_label, prob_values, label_test_matrix, test_metadata)

if type.strip('"\'') == 't':
    I_df = pd.DataFrame(index=metadata.names()[:-1], columns=metadata.names()[:-1])
    I_df = I_df.fillna(0)
    edge_weight_matrix = compute_edge_weights(feature_matrix, label_matrix, number_val_for_feature, metadata, I_df)
    adjacency_matrix = np.array(edge_weight_matrix)
    MST_parents = prim(adjacency_matrix)
    print_bayes_net_graph(MST_parents, metadata)
    conditional_probability = compute_conditional_probability(feature_matrix, label_matrix,
                                                              number_val_for_feature, MST_parents)

    test_feature_matrix, test_label_matrix, test_feature_range, test_metadata = read_data(test_file)
    class_labels, p_values = predict_bayes_net(test_feature_matrix, conditional_probability, test_feature_range,
                                               MST_parents)
    print_TAN_results(class_labels, p_values, test_label_matrix, test_metadata)
