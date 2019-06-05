import numpy as np
import csv

def load_wbc():
    
    MALIGNANT   =  1.0
    BENIGN      =  0.0
    PLACEHOLDER =  0.0
    
    FeatureTranslationDict = {        
        # Labels
        '4' : MALIGNANT,
        '2' : BENIGN
    }
    
    
    X = []
    y = []

    # 1. Open file
    with open('breast-cancer-wisconsin.data') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        # 2. Read each line and translate
        for row in reader:

            # 3. Translate features to floats
            features = []
            for i in range(1, 10):
                if row[i] == '?':
                    features.append(PLACEHOLDER)
                else:
                    features.append(int(row[i]))

            label = FeatureTranslationDict[row[10]]

            X.append(features)
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    
    return X, y