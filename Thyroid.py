import numpy as np
import csv

def load_thyroid():
    
    train_X = []
    train_y = []
    test_X  = []
    test_y  = []

    # Read training data
    with open('ann-train.data') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        
        for row in reader:

            features = []
            for i in range(len(row) - 3):
                features.append(float(row[i]))

            label = int(row[-3]) - 1

            train_X.append(features)
            train_y.append(label)

    train_X = np.array(train_X)
    train_y = np.array(train_y)
    
    
    # Read test data
    with open('ann-test.data') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        
        for row in reader:

            features = []
            for i in range(len(row) - 3):
                features.append(float(row[i]))

            label = int(row[-3]) - 1

            test_X.append(features)
            test_y.append(label)

    test_X = np.array(test_X)
    test_y = np.array(test_y)
    
    return train_X, train_y, test_X, test_y