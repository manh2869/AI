import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/home/w/AI/predict/Archive/Medicaldataset.csv")

X = df.drop(columns=["Result"])
y=df["Result"]

class GaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.means_ = None
        self.vars_ = None
        self.priors_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)                        
        n_classes = len(self.classes_)                      
        n_features = X.shape[1]                             

        self.means_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))      #           Age   HeartRate   BloodSugar
                                                            # Class 0    ?       ?           ?
                                                            # Class 1    ?       ?           ?
       
        self.priors_ = np.zeros(n_classes)                  #              class1   and   class2
                                                            # probability   P(+)           P(-)

        for i, c in enumerate(self.classes_):               #       self.classes_ =[+,-]
            X_c = X[y == c]                                 #       c  =  class   , i = index
            
            self.means_[i] = X_c.mean(axis=0)                           #    self.mean=[  ? ,   ?  ,   ?]
            self.vars_[i] = X_c.var(axis=0) + self.var_smoothing        
                                                                        
                                                                        #    self.means_[i] =  (mean class c )
            
            self.priors_[i] = X_c.shape[0] / X.shape[0]                 #    sample class c /  sample data
           

        return self

    def _log_likelihood(self, X):                
                                                                        #             class 0    class 1
        n_classes = len(self.classes_)                                  #   people 1│   0     │    0    │
        n_samples = X.shape[0]                                          #   people 2│   0     │    0    │
        log_proba = np.zeros((n_samples, n_classes))                    #   people 3│   0     │    0    │

        for i in range(n_classes):
            diff = X - self.means_[i]                                   #        X  -   means
            log_prob_features = (                                       #        log[ 1/sqrt(2*pi*sigma^2) * exp(-(x-mu)^2/(2*sigma^2)) ]
                -0.5 * np.log(2 * np.pi * self.vars_[i])                #                  Age      |     Heart rate
                                                                        #  poeple 1       -3.1      |      -2.0
                - 0.5 * (diff**2) / self.vars_[i]                       
            )
            log_proba[:, i] = log_prob_features.sum(axis=1) + np.log(self.priors_[i])       #  sum class of people i 
                                                                                            #  logP(Age∣C)+logP(HeartRate∣C....+ n_feature)
                                                                                            #  multiplication by the priors
        return log_proba                                       

    def predict(self, X):
        log_proba = self._log_likelihood(X)
        return self.classes_[np.argmax(log_proba, axis=1)]              #  row 1: [-5.8, -3.2]
                                                                        #           ↑       ↑
                                                                        #        class 0 class 1
                                                                        # max = -3.2

    def predict_proba(self, X):
        log_proba = self._log_likelihood(X)
        log_proba -= log_proba.max(axis=1, keepdims=True)
        proba = np.exp(log_proba)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)


#'Age', 'Gender', 'Heart rate', 'Systolic blood pressure' , 'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin',
people = [
    [28,  0,  72, 115, 75,  90,  1.2,  0.003],   # negative
    [50,  1,  85, 135, 80, 180,  3.1,  0.012],   # negative
    [62,  1,  95, 160, 95, 240, 18.5,  1.200],   # positive
    [70,  1, 102, 175, 98, 310, 85.0,  3.500],   # positive
    [65,  0,  88, 155, 92, 260, 12.4,  0.850],   # positive
]

for i, p in enumerate(people, 1):
    arr = np.array([p])
    pred = model.predict(arr)[0]
    prob = model.predict_proba(arr)[0]
    print(f"people {i}: {pred}  (negative={prob[0]*100:.1f}%, positive={prob[1]*100:.1f}%)")
