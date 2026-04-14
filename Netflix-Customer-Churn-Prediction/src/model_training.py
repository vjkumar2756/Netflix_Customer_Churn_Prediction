from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def train_logistic(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(X_train, y_train)
    return model