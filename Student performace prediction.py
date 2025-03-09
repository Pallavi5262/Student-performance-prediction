import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import sklearn.utils as shuffle
import sklearn.preprocessing as pp
import sklearn.tree as tr
import sklearn.ensemble as es
import sklearn.metrics as metrics
import sklearn.linear_model as lm
import sklearn.neural_network as nn
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_csv("New_AI_Data.csv")

# Drop unnecessary columns
columns_to_drop = ["gender", "StageID", "GradeID", "NationalITy", "PlaceofBirth",
                   "SectionID", "Topic", "Semester", "Relation", "ParentschoolSatisfaction",
                   "ParentAnsweringSurvey", "AnnouncementsView"]
data.drop(columns=columns_to_drop, inplace=True)

# Shuffle dataset
data = shuffle.shuffle(data)

# Encode categorical variables
for col in data.select_dtypes(include=['object']).columns:
    data[col] = pp.LabelEncoder().fit_transform(data[col])

# Split dataset into features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
train_size = int(0.7 * len(data))

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define models
models = {
    "Decision Tree": tr.DecisionTreeClassifier(),
    "Random Forest": es.RandomForestClassifier(),
    "Perceptron": lm.Perceptron(),
    "Logistic Regression": lm.LogisticRegression(),
    "MLP Classifier": nn.MLPClassifier(activation="logistic")
}

# Train models and evaluate performance
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {round(accuracy, 3)}")
    print(metrics.classification_report(y_test, y_pred))

# Graph Visualization
def plot_graph(x_col, title, order=None):
    print(f"\nLoading {title}...\n")
    time.sleep(1)
    plt.figure(figsize=(10, 6))
    sns.countplot(x=x_col, hue='Class', data=pd.read_csv("AI-Data.csv"), order=order)
    plt.title(title)
    plt.show()

menu = {
    1: ("Marks Class Count Graph", "Class", ['L', 'M', 'H']),
    2: ("Marks Class Semester-wise Graph", "Semester", None),
    3: ("Marks Class Gender-wise Graph", "gender", ['M', 'F']),
    4: ("Marks Class Nationality-wise Graph", "NationalITy", None),
    5: ("Marks Class Grade-wise Graph", "GradeID", ['G-02', 'G-04', 'G-05', 'G-06', 'G-07', 'G-08', 'G-09', 'G-10', 'G-11', 'G-12']),
    6: ("Marks Class Section-wise Graph", "SectionID", None),
    7: ("Marks Class Topic-wise Graph", "Topic", None),
    8: ("Marks Class Stage-wise Graph", "StageID", None),
    9: ("Marks Class Absent Days-wise Graph", "StudentAbsenceDays", None),
    10: ("Exit", None, None)
}

choice = 0
while choice != 10:
    print("\n".join([f"{k}. {v[0]}" for k, v in menu.items()]))
    choice = int(input("Enter Choice: "))

    if choice in menu and choice != 10:
        plot_graph(menu[choice][1], menu[choice][0], menu[choice][2])
    elif choice == 10:
        print("Exiting...")
        time.sleep(1)

# Custom Input Prediction
if input("Do you want to test specific input (y/n): ").lower() == "y":
    features = []
    features.append(int(input("Enter Raised Hands: ")))
    features.append(int(input("Enter Visited Resources: ")))
    features.append(int(input("Enter Number of Discussions: ")))
    features.append(int(input("Enter Absenteeism (0 for Above-7, 1 for Under-7): ")))

    user_input = np.array(features).reshape(1, -1)
    class_map = {0: "H", 1: "M", 2: "L"}

    for name, model in models.items():
        pred = class_map[model.predict(user_input)[0]]
        print(f"{name} Prediction: {pred}")

print("\nExiting...")
time.sleep(1)