#Krishna Malhotra
#GitHub-KriMal-15

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random

import joblib

#Models
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

filename = "Gesture_Classifier_Model.joblib"

data0 = pd.read_csv("data/0.csv", header=None) # For class 0: rock
data1 = pd.read_csv("data/1.csv", header=None) # For class 1: scissors
data2 = pd.read_csv("data/2.csv", header=None) # For class 2: paper

# 8 consecutive readings of all 8 sensors which is why 64 columns plus last column is the class = 65 columns

# Now, we will combine all the dataset into 1 big dataset
data = pd.concat([data0,data1,data2], axis=0)
#print("Model Size:", data.shape)

Y = data.iloc[:,-1]
X = data.drop(data.columns[-1], axis=1)

# Now, train test split
X_train, Xtest, Y_train, Ytest = train_test_split(X, Y, train_size=0.8, random_state=10)

rf = SVC(random_state=100, C=1, kernel="rbf")
y_pred = rf.fit(X_train, Y_train).predict(Xtest)
f1_svc = f1_score(Ytest, y_pred, average='micro')
print("Support Vector Classifier Model Accuracy:", f1_svc * 100, "%")
print("")
print("")
#ROCK PAPER SCISSORS game
def determineGesture(val):
    if val == 0:
        #print("Predicted Gesture is ROCK")
        return "ROCK"
    if val == 1:
        #print("Predicted Gesture is SCISSORS")
        return "Scissors"
    if val == 2:
        #print("Predicted Gesture is PAPER")
        return "Paper"
def Game(user_action_AI, computer_action):

    if user_action_AI == computer_action:
        print("It's a tie!")
    elif user_action_AI == 0:
        if computer_action == 1:
            print("Rock smashes scissors! You win!")
        else:
            print("Paper covers rock! You lose.")
    elif user_action_AI == 2:
        if computer_action == 0:
            print("Paper covers rock! You win!")
        else:
            print("Scissors cuts paper! You lose.")
    elif user_action_AI == 1:
        if computer_action == 2:
            print("Scissors cuts paper! You win!")
        else:
            print("Rock smashes scissors! You lose.")

joblib.dump(rf, filename)
# load model
loaded_model = joblib.load(filename)

testArrayRock = [[-26.0,-7.0,-6.0,-8.0,-13.0,4.0,24.0,12.0,23.0,0.0,3.0,9.0,-7.0,-6.0,-16.0,-2.0,-24.0,-2.0,4.0,5.0,37.0,-3.0,9.0,-7.0,17.0,-1.0,-5.0,4.0,-13.0,9.0,-3.0,3.0,-18.0,2.0,5.0,1.0,-3.0,2.0,-8.0,-9.0,21.0,0.0,2.0,5.0,5.0,-3.0,46.0,28.0,-35.0,-1.0,0.0,-13.0,-9.0,-8.0,-90.0,-44.0,44.0,1.0,-4.0,-3.0,-1.0,-2.0,66.0,18.0]]
testArrayScissors = [[3.0,-1.0,3.0,3.0,1.0,-14.0,-1.0,1.0,-18.0,1.0,-2.0,-1.0,-14.0,-25.0,-4.0,-4.0,43.0,4.0,1.0,0.0,7.0,6.0,3.0,6.0,3.0,3.0,1.0,2.0,-3.0,-4.0,0.0,3.0,-2.0,1.0,-6.0,-7.0,58.0,48.0,2.0,-11.0,12.0,3.0,9.0,12.0,-40.0,-63.0,-5.0,8.0,11.0,5.0,2.0,1.0,-38.0,-55.0,-1.0,6.0,-8.0,-1.0,-1.0,-4.0,5.0,-8.0,1.0,1.0]]
testArrayPaper = [[4.0,19.0,-9.0,-7.0,-3.0,-36.0,-6.0,-23.0,3.0,-21.0,-2.0,-9.0,15.0,-23.0,-11.0,-2.0,11.0,27.0,-3.0,-12.0,-22.0,-34.0,-16.0,-2.0,-10.0,-9.0,3.0,5.0,41.0,-33.0,19.0,1.0,5.0,0.0,-2.0,-6.0,-12.0,63.0,-7.0,-3.0,-11.0,-15.0,4.0,12.0,42.0,12.0,-14.0,-20.0,1.0,29.0,-2.0,-7.0,-24.0,-22.0,-8.0,9.0,-14.0,-2.0,-3.0,-4.0,-21.0,7.0,-8.0,-12.0]]

y_predicted = loaded_model.predict(testArrayRock)

predGesture = determineGesture(y_predicted)
comp_generated_move = random.randint(0,2)
print("Your move is: ", predGesture)
print("Computer action is: ", determineGesture(comp_generated_move))

print("")

Game(y_predicted, comp_generated_move)

#0 - rock
#1 = scissors
#2 = paper






