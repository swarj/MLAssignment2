import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input = pd.read_csv(dataFile)

    def preprocess(self):
        self.df = self.raw_input
        toDrop = ['No']
        self.df = self.df.drop(toDrop, axis = 1)
        self.df = self.df.dropna(axis=1)
        self.df = (self.df - self.df.mean())/ self.df.std()
        self.df.style
   

    def train_evaluate(self):
        ncols = len(self.df.columns)
        nrows = len(self.df.index)
        X = self.df.iloc[:, 0:(ncols - 1)]
        y = self.df.iloc[:, (ncols-1)]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y)

        model = Sequential()
        model.add(Dense(4, activation='relu', input_dim=X_train.shape[1]))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        adam = Adam(learning_rate=0.1)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=100, batch_size = 1)
        #activations = ['logistic', 'tanh', 'relu']
        #learning_rate = [0.01, 0.1]
        #max_iterations = [100, 200] # also known as epochs
        #num_hidden_layers = [2, 3]
        
        return 0


if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/swarj/MLAssignment2/main/slump_test%20(1).csv"
    neural_network = NeuralNet(url) # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()