import numpy as np
import csv

def load_tic_tac_toe():
    
    B = 0.0
    O = 1.0
    X = 2.0
    
    NEGATIVE = 0.0
    POSITIVE = 1.0
    
    FeatureTranslationDict = {
        
        # Features
        'b': B,
        'o': O,
        'x': X,
        
        # Labels
        'negative': NEGATIVE,
        'positive': POSITIVE
    }
    
    
    X = []
    y = []

    # 1. Open file
    with open('tic-tac-toe.data') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        # 2. Read each line and translate
        for row in reader:

            # 3. Translate features to floats
            features = []
            for i in range(9):
                feature = FeatureTranslationDict[row[i]]
                features.append(feature)

            label = FeatureTranslationDict[row[9]]

            X.append(features)
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    
    return X, y