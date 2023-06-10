import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load data
folder_path = "./bbcsport-fulltext/bbcsport/football"
X = []
y = []
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()
        X.append(data)
        y.append(filename.split(".")[0])  # use file name as label

# Vectorize data
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Train model
clf = LinearSVC()
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model as .pkl file
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)

    # open the model
    # import pickle

    # with open('model.pkl', 'rb') as f:
    # clf = pickle.load(f)
