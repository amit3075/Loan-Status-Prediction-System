# import operator
# from collections import Counter
#
# class KNearestNeighbors:
#     def __init__(self,k):
#         self.k = k
#
#     def fit(self, X_train, y_train):
#         self.X_train = X_train
#         self.y_train = y_train
#         print("training completed")
#
#     def predict(self,X_test):
#
#         distance ={}
#         counter =1
#
#         for i in self.X_train:
#             distance[counter] =((X_test[0][0]-i[0])**2 + (X_test[0][1]-i[1])**2)**1/2
#             counter = counter +1
#         # print(distance)
#         distance= sorted(distance.items(),key=operator.itemgetter(1))
#
#         # print(distance)
#         self.classify(distance=distance[:self.k])
#     def classify(self,distance):
#         label=[]
#         for i in distance:
#             label.append(self.y_train[i[0]])
#
#         # print(Counter(label))
#         return ((Counter(label).most_common()[0][0]))
#
from collections import Counter
import operator
import random

import numpy as np


# Define the range of k values to consider
# k_min = 1
# k_max = 10
#
# # Generate a random value of k
# k = random.randint(k_min, k_max)
class KNearestNeighbors:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        print("Training completed")


    def predict(self, X_test_list):
        distances = {i: [] for i in range(len(X_test_list))}
        for i, X_test in enumerate(X_test_list):
            for j, X_train in enumerate(self.X_train):
                dist = 0
                for k in range(len(X_test)):
                    dist += (X_test[k] - X_train[k]) ** 2
                dist = dist ** 0.5
                distances[i].append((j, dist))
            distances[i] = sorted(distances[i], key=operator.itemgetter(1))
        return self.classify(distances)

    def classify(self, distances):
        labels = []
        for i in range(len(distances)):
            k_neighbors = distances[i][:self.k]
            k_labels = [self.y_train[j] for j, dist in k_neighbors]
            label = Counter(k_labels).most_common(1)[0][0]
            labels.append(label)
        return labels


    # def predict_new(X_new):
    #     # knn = joblib.load('knn_model.joblib')
    #     # scaler = joblib.load('scaler.joblib')
    #     X_new = np.array(X_new).reshape(1, 4)
    #     X_new = scaler.transform(X_new)
    #     result = knn.predict(X_new)
    #     if result == 0:
    #         print("Loan application will not be approved")
    #     else:
    #         print("Loan application will be approved")
    # #
