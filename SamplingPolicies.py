import numpy as np

class Point:
    
    def __init__(self, X, y):
        
        self.X = X
        self.y = y

        
class UniformSamplingPolicy:
    
    def __init__(self, X, y):
        self.points = []
        
        for i in range(len(X)):
            self.points.append(Point(X[i], y[i]))
            
        self.points = np.array(self.points)
        
    def sample(self, num_samples):
        samples = np.random.choice(self.points,
                                   size = num_samples,
                                   replace = False)

        return samples
        

class ClassUniformSamplingPolicy:

    def __init__(self, X, y):
        
        # Record number of classes
        self.num_classes = np.max(y) + 1
        
        # List of points is indexed as:
        #
        #    self.points[class_number][point_index]
        #
        self.points = []
        
        # Create num_classes sub-lists
        for i in range(self.num_classes):
            self.points.append([])
        
        # Create master list of points, indexed by class, to sample from
        for i, x in enumerate(X):
            self.points[y[i]].append(Point(x, y[i]))
            
        self.points = np.array(self.points)


    def sample(self, num_points):
        
        # Final array of samples to return
        samples = []
        
        num_points_per_class = []
        for i in range(self.num_classes):
            num_points_per_class.append(num_points // self.num_classes)
        
        remainder = num_points % self.num_classes
        for i in range(remainder):
            num_points_per_class[i] += 1
        
        for i in range(self.num_classes):
            
            # If we are asking for more points than are in the population,
            # then we need to oversample. Otherwise, do not replace after
            # each selection
            do_replace = False
            if num_points_per_class[i] > len(self.points[i]):
                do_replace = True
            
            class_samples = np.random.choice(self.points[i],
                                             size=num_points_per_class[i],
                                             replace=do_replace)
            
            samples += class_samples.tolist()
            
        # Shuffle and return samples
        samples = np.array(samples)
        np.random.shuffle(samples)
        return samples
            