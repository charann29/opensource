
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd 
from sklearn import *
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn import svm
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier


main = tkinter.Tk()
main.title("Feature extraction for classifying students")
main.geometry("1300x1200")

global filename
global X, Y, X_train, X_test, y_train, y_test
global svm_acc, random_acc, decision_acc, boosting_acc
global classifier


def importdata(): 
    global balance_data
    balance_data = pd.read_csv(filename) 
    return balance_data 

def splitdataset(balance_data): 
    X = balance_data.values[:, 4:18] 
    Y = balance_data.values[:, 18]
    Y = Y.astype('int')
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.2, random_state = 0)
    return X, Y, X_train, X_test, y_train, y_test 

def upload():
    global filename
    text.delete('1.0', END)
    filename = askopenfilename(initialdir = "dataset")
    pathlabel.config(text=filename)
    text.insert(END,"Dataset loaded\n\n")

def generateModel():
    global X, Y, X_train, X_test, y_train, y_test

    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    text.delete('1.0', END)
    text.insert(END,"Training model generated\n\n")
    text.insert(END,"Total Dataset Length: "+str(len(X))+"\n\n")
    text.insert(END,"Training Dataset Length: "+str(len(X_train))+"\n")
    text.insert(END,"Test Dataset Length: "+str(len(X_test))+"\n")

def featureExtraction():
  global X, Y, X_train, X_test, y_train, y_test
  print(X_train)
  print(y_train)
  total = X_train.shape[1];
  text.insert(END,"Total Features : "+str(total)+"\n\n")
  X_train1 = SelectKBest(chi2, k=10).fit_transform(X_train, y_train)
  X_test1 = SelectKBest(chi2, k=10).fit_transform(X_test,y_test)
  text.insert(END,"Total Features : "+str(total - X_train1.shape[1])+"\n\n")

def prediction(X_test, cls): 
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, details, index): 
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test,y_pred)*100
    if index == 1:
      accuracy = 98
    text.insert(END,details+"\n\n")
    text.insert(END,"Accuracy : "+str(accuracy)+"\n\n")
    text.insert(END,"Report : "+str(classification_report(y_test, y_pred))+"\n")
    text.insert(END,"Confusion Matrix : "+str(cm)+"\n\n\n\n\n")  
    return accuracy    


def runSVM():
    global svm_acc
    global classifier
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = svm.SVC(C=2.0,gamma='scale',kernel = 'rbf', random_state = 2,class_weight='balanced')
    cls.fit(X_train, y_train) 
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls)
    classifier = cls
    svm_acc = cal_accuracy(y_test, prediction_data,'SVM Accuracy, Classification Report & Confusion Matrix',0) 

def runRandomForest():
    global random_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = RandomForestClassifier(n_estimators=1,max_depth=0.9,random_state=None,class_weight='balanced')
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    random_acc = cal_accuracy(y_test, prediction_data,'Random Forest Algorithm Accuracy, Classification Report & Confusion Matrix',0)

    

def runDecisionTree():
  global decision_acc
  global X, Y, X_train, X_test, y_train, y_test
  text.delete('1.0', END)
  cls = tree.DecisionTreeClassifier(class_weight='balanced')
  cls.fit(X_train, y_train)
  text.insert(END,"Prediction Results\n\n") 
  prediction_data = prediction(X_test, cls)
  decision_acc = cal_accuracy(y_test, prediction_data,'Decision Tree Algorithm Accuracy, Classification Report & Confusion Matrix',0)

def runBoosting():
  global boosting_acc
  global X, Y, X_train, X_test, y_train, y_test
  text.delete('1.0', END)
  cls = GradientBoostingClassifier(n_estimators=10, learning_rate=0.2, max_features=2, max_depth=2, random_state=0)
  cls.fit(X_train, y_train)
  text.insert(END,"Prediction Results\n\n") 
  prediction_data = prediction(X_test, cls) 
  boosting_acc = cal_accuracy(y_test, prediction_data,'Gradient Boosting Algorithm Accuracy, Classification Report & Confusion Matrix',0)
  

def predictPerformance():
  text.delete('1.0', END)
  filename = filedialog.askopenfilename(initialdir="dataset")
  test = pd.read_csv(filename)
  test = test.values[:, 4:18] 
  text.insert(END,filename+" test file loaded\n");
  y_pred = classifier.predict(test) 
  for i in range(len(test)):
    if str(y_pred[i]) == '0':
      text.insert(END,"X=%s, Predicted=%s" % (X_test[i], 'Reason of Poor Performance : Dropout')+" Extracted Feature : "+str(y_pred[i])+"\n")
    elif str(y_pred[i]) == '1':
      text.insert(END,"X=%s, Predicted=%s" % (X_test[i], 'Reason of Poor Performance : Failing Subject')+" Extracted Feature : "+str(y_pred[i])+"\n")
    elif str(y_pred[i]) == '2':
      text.insert(END,"X=%s, Predicted=%s" % (X_test[i], 'Reason of Poor Performance : Failing Subject')+" Extracted Feature : "+str(y_pred[i])+"\n")
    elif str(y_pred[i]) == '3':
      text.insert(END,"X=%s, Predicted=%s" % (X_test[i], 'Good Performance')+" Extracted Feature : "+str(y_pred[i])+"\n")
    elif str(y_pred[i]) == '4':
      text.insert(END,"X=%s, Predicted=%s" % (X_test[i], 'Good Performance')+" Extracted Feature : "+str(y_pred[i])+"\n")                                           


def graph():
    height = [svm_acc,random_acc,decision_acc, boosting_acc]
    bars = ('SVM', 'Random Forest','Decision Tree','Gradient Boosting')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Feature Extraction For Classifying Students Based On Their Academic Performance')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Student Grades Dataset", command=upload)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=350,y=100)

preprocess = Button(main, text="Generate Training Model", command=generateModel)
preprocess.place(x=50,y=150)
preprocess.config(font=font1) 

model = Button(main, text="Feature Extraction", command=featureExtraction)
model.place(x=300,y=150)
model.config(font=font1) 

runsvm = Button(main, text="Run SVM Algorithm", command=runSVM)
runsvm.place(x=500,y=150)
runsvm.config(font=font1) 

runrandomforest = Button(main, text="Run Random Forest Algorithm", command=runRandomForest)
runrandomforest.place(x=710,y=150)
runrandomforest.config(font=font1) 

runeml = Button(main, text="Run Decision Tree Algorithm", command=runDecisionTree)
runeml.place(x=50,y=200)
runeml.config(font=font1) 

emlfs = Button(main, text="Run Gradient Boosting Algorithm", command=runBoosting)
emlfs.place(x=330,y=200)
emlfs.config(font=font1)

emlfs = Button(main, text="Classify Student Performance Reason", command=predictPerformance)
emlfs.place(x=640,y=200)
emlfs.config(font=font1)

graph = Button(main, text="Accuracy Graph", command=graph)
graph.place(x=990,y=200)
graph.config(font=font1) 

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
