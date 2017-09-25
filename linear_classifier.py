import numpy as np

def clean_datafile(datafile):
    data = ''
    with open(datafile) as fp:
        data = fp.read().strip()

    X, Y = [], []
    classification_dict = {}
    count = 1  # classification value conversion
    for line in data.split():
        temp = line.strip().split(',')
        if not temp[-1] in classification_dict:
            classification_dict[temp[-1]] = count
            count += 1
        Y.append(classification_dict[temp[-1]]) 
        temp = [1] + list(map(float, temp[:-1]))
        X.append(temp)
    print (classification_dict)
    return (X, Y, count-1)

def train_linear_classifier(X, Y):
    X_arr = np.array(X)
    Y_arr = np.array(Y)
    beta_values = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X_arr), X_arr)), np.transpose(X_arr)), Y_arr)
    return beta_values

def test_linear_classifier(beta, test_data, test_classes, num_classes):
    predicted_y = np.dot(beta, np.transpose(test_data))
    # rounding off values;
    predicted_y = [round(i) for i in predicted_y] 
    true_positives, false_positives, true_negatives, false_negatives = 0, 0, 0, 0
    for i in range(1, num_classes+1): 
        # true positive means correctly identifying 'cat' as 'cat' 
        true_positives += len([j for j in range(len(test_data)) if (test_classes[j] == predicted_y[j]) and 
                                                                     (predicted_y[j] == i)])
        # false positive means incorrectly identifying 'non cat' as 'cat'
        temp_list = [j for j in range(1, num_classes+1) if j != i]
        false_positives += len([j for j in range(len(test_data)) if test_classes[j] != predicted_y[j] and
                                                                      (predicted_y[j] in temp_list)])
        # true negative means correctly identifying 'non cat' as 'non cat'
        temp_list = [j for j in range(1, num_classes+1) if j != i]
        true_negatives += len([j for j in range(len(test_data)) if test_classes[j] == predicted_y[j] and
                                                                      (predicted_y[j] in temp_list)])
        # false negative means incorrectly identifying 'cat' as 'non cat'
        false_negatives += len([j for j in range(len(test_data)) if (test_classes[j] != predicted_y[j]) and
                                                                    (predicted_y[j] == i)])
    true_positives, false_positives, true_negatives, false_negatives = true_positives / num_classes, false_positives / num_classes, true_negatives / num_classes, false_negatives / num_classes
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    return accuracy

def cross_validation(X, Y, num_folds, num_classes):
    subset_size = len(X) // num_folds
    # push the biggest subset to the end if the subsets are not equal in size
    avg_accuracy = 0
    for i in range(num_folds):
        if i == (num_folds - 1) and (i+1)*subset_size < len(X):
            test_data = X[i*subset_size:]
            test_classes = Y[i*subset_size:]
            training_data = X[:i*subset_size]
            training_classes = Y[:i*subset_size]
        else:
            test_data = X[i*subset_size:(i+1)*subset_size]
            test_classes = Y[i*subset_size:(i+1)*subset_size]
            training_data  = X[:i*subset_size] + X[(i+1)*subset_size:]
            training_classes = Y[:i*subset_size] + Y[(i+1)*subset_size:]
        beta = train_linear_classifier(training_data, training_classes)
        #print (beta)
        accuracy_values = test_linear_classifier(beta, test_data, test_classes, num_classes)
        #for j in accuracy_values:
        #    avg_accuracy[j-1] += accuracy_values[j] 
        print ('Accuracy values for fold', i, 'are', accuracy_values)
    #avg_accuracy = [j/num_folds for j in avg_accuracy]
        avg_accuracy += accuracy_values 
    avg_accuracy = avg_accuracy / num_folds
    print (avg_accuracy)

if __name__ == '__main__':
    datafile = input('Enter the name of the file to be tested : ')
    X, Y, num_classes = clean_datafile(datafile)
    print ('num classes', num_classes)
    cross_validation(X, Y, 15, num_classes)
