import numpy as np
import csv

def load_statlog():
    
    train_X = []
    train_y = []
    test_X  = []
    test_y  = []

    # Read training data
    with open('shuttle.trn') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        
        for row in reader:

            features = []
            for i in range(len(row) - 1):
                features.append(float(row[i]))

            label = int(row[-1]) - 1

            train_X.append(features)
            train_y.append(label)

    train_X = np.array(train_X)
    train_y = np.array(train_y)
    
    
    # Read test data
    with open('shuttle.tst') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        
        for row in reader:

            features = []
            for i in range(len(row) - 1):
                features.append(float(row[i]))

            label = int(row[-1]) - 1

            test_X.append(features)
            test_y.append(label)

    test_X = np.array(test_X)
    test_y = np.array(test_y)
    
    return train_X, train_y, test_X, test_y, 9, 7