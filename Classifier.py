# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 00:35:18 2020

@author: Administrador
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense

class Classifier():
    
    def __init__(self):
        pass
    def train(self):
        
        dataset = pd.read_csv("/Users/Administrador/desafio_fraude.csv", header = None)
        
        data_array = dataset.to_numpy()
        
        data_array = data_array[1:,1:]
        
        data_array = data_array.astype(float)
        
        scaled_array = data_array.copy()
        
        print('data_array type: '+str(type(data_array)))
        
        print('scaled_array type: '+str(type(scaled_array)))
        
        data_array, scaled_array = self.scale_array(data_array, scaled_array)
        
        print('data before shape: '+str(data_array.shape))
        
        data_array, y_train = self.create_y(data_array)
        
        print('data array shape: '+str(data_array.shape))
        print('y_train shape: '+str(y_train.shape))
        
        concat_array = np.concatenate((data_array, y_train),axis=1)
        
        print('concat array shape: '+str(concat_array.shape))
        
        #raise ValueError('print')
        

        
        
        # Test class ratio:
        class_ratio = self.get_ratio(y_train)
        
        # Fix class imbalance:
        concat_array = self.balance_classes(class_ratio, concat_array)
        
        np.save('concat_array.npy', concat_array)
        print('concat_array saved')
        
        try:
            print('y_train shape: '+str(y_train.shape))
        except:
            print('y_train len: '+str(len(y_train)))
        
        data_array, y_train = self.create_y(concat_array)
        
        # Test the class ratio after balancing:
        class_ratio = self.get_ratio(y_train)
        print('class ratio after balancing: '+str(class_ratio))
        
        input_size = 29
        
        # Test which columns are more important, by deleting one column at a time:
        #input_size = 28
        #for i in reversed(range(20)):
            #data_array = np.delete(data_array, i, 1)
        #data_array = np.delete(data_array, 28, 1)

                    
        X_train, X_test, y_train, y_test = train_test_split(data_array, y_train, test_size = 0.2, random_state = 0)
        
        print('data array shape: '+str(data_array.shape))
        
        # save y_test for future accuracy tests:
        np.save('y_test.npy', y_test)
        print('y_test saved')
        
        #raise ValueError('save')
        
        
        
        
        # Initialising the ANN
        classifier = Sequential()
        
        # Adding the input layer and the first hidden layer
        classifier.add(Dense(units = 400, kernel_initializer = 'uniform', activation = 'relu', input_dim = input_size))
        
        # Adding the second hidden layer
        classifier.add(Dense(units = 400, kernel_initializer = 'uniform', activation = 'relu'))
        
        # Adding the second hidden layer
        classifier.add(Dense(units = 400, kernel_initializer = 'uniform', activation = 'relu'))
        
        # Adding the output layer
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        
        # Compiling the ANN
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',)
        
        
        # Fitting the ANN to the Training set
        classifier.fit(X_train, y_train, batch_size = 64, epochs = 30)
        
        self.predict()
        
    def predict(self):
    
        # Part 3 - Making predictions and evaluating the model
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        np.save('y_pred.npy', y_pred)
        print('predictions saved')
        #y_pred = (y_pred > 0.5)
        
        self.accuracy()
        
    def accuracy(self):
    
        # Calculate Accuracy:
        y_pred = np.load('y_pred.npy')
        y_test = np.load('y_test.npy')
        
        y_comparison = np.concatenate((y_test, y_pred),axis=1)
        
        accuracy = self.calculate_accuracy(y_comparison)
        
        print('accuracy: '+str(accuracy)+' %')
        
        # Making the Confusion Matrix
        #from sklearn.metrics import confusion_matrix
        #cm = confusion_matrix(y_test, y_pred)
        
    def scale_array(self, data_array, scaled_array):
        
        for column in range(len(data_array[0])):
            
            numbers = data_array[:,column]
            max_c = max(data_array[:,column])
            min_c = min(data_array[:,column])
            diff_c = max_c - min_c
            
            min_0 = 0 - min_c
            
            scaled = numbers + min_0
            
            scaled = scaled/(max_c + min_0)
            
            for i in range(len(scaled_array[:,column])):
                scaled_array[i,column] = scaled[i]
        
        data_array = scaled_array.copy()
        
        return data_array, scaled_array
    
    def create_y(self, data_array):
            
        y_train = data_array[:,-1]
        
        data_array = data_array[:,:-1]
        
        y_train = np.expand_dims(y_train, axis=1)
        
        return data_array, y_train
            
    def get_ratio(self, y_train):
        positive_count = 0
        negative_count = 0
        for row in y_train:
            if row == 1:
                positive_count +=1
                
            elif row == 0:
                negative_count +=1
                
        class_ratio = negative_count / positive_count
        #print(class_ratio)
            
        class_ratio = int(class_ratio)
        
        return class_ratio
    
    def balance_classes(self, class_ratio, concat_array):
        # If balanced array is already saved, load it:
        try:
            concat_array = np.load('concat_array.npy')
        # Else, make it
        except:
            # Solve class imbalance by replicating the smallest class
            new_bool = False
            for row in range(len(concat_array)):
                
                print(str(row)+', '+str(concat_array[row][-1]))
                #print('class_ratio: '+str(class_ratio))
                #raise ValueError('print')
                if concat_array[row][-1] == 1:
                    
                    for copy in range(class_ratio -1):
                        if new_bool == False:
                            new_array = concat_array[row]
                            new_bool = True
                        else:
                            new_array = np.vstack([new_array, concat_array[row]])
            
            print('shape concat before joining: '+str(concat_array.shape))
            print('shape new_array: '+str(new_array.shape))
            concat_array = np.vstack([concat_array, new_array])            
            np.random.shuffle(concat_array)
            print('len concat after joining: '+str(len(concat_array)))

        return concat_array
    
    def calculate_accuracy(self, y_comparison):
        right_count = 0
        for i in range(len(y_comparison)):
            
            if round(y_comparison[i][0]) == round(y_comparison[i][1]):
                right_count +=1
        accuracy = right_count/len(y_comparison)*100
        return accuracy
        

classifier_class = Classifier()
classifier_class.train()
classifier_class.predict()
classifier_class.accuracy()
