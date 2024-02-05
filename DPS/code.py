import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# load dataset
data = pd.read_csv('diabetes.csv')
l1 = data['Outcome'].unique().tolist()

# variables
Name = tk.StringVar()
Symptom1 = tk.StringVar()
Symptom2 = tk.StringVar()
Symptom3 = tk.StringVar()
Symptom4 = tk.StringVar()
Symptom5 = tk.StringVar()

# functions
def DecisionTree():
    # split data into features and target
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # create DecisionTreeClassifier
    clf = DecisionTreeClassifier()

    # train the model
    clf.fit(X_train, y_train)

    # make predictions
    y_pred = clf.predict(X_test)

    # calculate accuracy
    acc = accuracy_score(y_test, y_pred)

    # display accuracy
    t1.delete(1.0, tk.END)
    t1.insert(tk.END, f"Accuracy: {acc}")

def randomforest():
    # split data into features and target
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # create RandomForestClassifier
    clf = RandomForestClassifier()

    # train the model
    clf.fit(X_train, y_train)

    # make predictions
    y_pred = clf.predict(X_test)

    # calculate accuracy
    acc = accuracy_score(y_test, y_pred)

    # display accuracy
    t2.delete(1.0, tk.END)
    t2.insert(tk.END, f"Accuracy: {acc}")

def NaiveBayes():
    # split data into features and target
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # create GaussianNB
    clf = GaussianNB()

    # train the model
    clf.fit(X_train, y_train)

    # make predictions
    y_pred = clf.predict(X_test)

    # calculate accuracy
    acc = accuracy_score(y_test, y_pred)

    # display accuracy
    t3.delete(1.0, tk.END)
    t3.insert(tk.END, f"Accuracy: {acc}")

# create tkinter window
root = tk.Tk()
root.title("Diabetes Prediction")

# labels
NameLb = tk.Label(root, text="Name", fg="yellow", bg="black")
NameLb.grid(row=6, column=0, pady=15, sticky=W)

S1Lb = tk.Label(root, text="Symptom 1", fg="yellow", bg="black")
S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = tk.Label(root, text="Symptom 2", fg="yellow", bg="black")
S2L