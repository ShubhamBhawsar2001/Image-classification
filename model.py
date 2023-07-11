import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None

    
   
    def fit(self, X) -> None:
        # fit the PCA model
        X_new=X
        mean=np.mean(X,axis=0)
        X_new=X_new-mean
        cov=np.dot(X_new.transpose(),X_new)
        eignval,eignvector = np.linalg.eig(cov)
        
        value=np.argsort(eignval)[::-1]
        sort_eignvector=eignvector[:,value]
        self.components=sort_eignvector[:,:self.n_components]
        
    def transform(self, X) -> np.ndarray:
        
        return np.dot(X,self.components)
        
    def fit_transform(self, X) -> np.ndarray:
        # fit the model and transform the data
        self.fit(X)
        return self.transform(X)

class SupportVectorModel:
    def __init__(self) -> None:
        self.w = None
        self.b = None
    
    def _initialize(self, X) -> None:
        # initialize the parameters 
        
        self.w=np.random.uniform(0,1,size=len(X[0]))
        self.b=0.1
        
        


    def fit(
            self, X, y, 
            learning_rate: float,
            num_iters: int,
            C: float = 1.0,
    ) -> None:
        self._initialize(X)
        
        #epoch=[]
        #loss_history = []
        # fit the SVM model using stochastic gradient descent
        for j in tqdm(range(1, num_iters + 1)):
            n=np.size(y)#60000
            random= np.random.randint(0, n)
            datapoint=X[random]

            distance = 1 - y[random] *(np.dot(self.w,datapoint)+self.b)
            dw = 0
            db=0
            
            if max(0, distance) != 0:
                dw = self.w-C*y[random]*datapoint
                db=-C*y[random]
            else:
                dw = self.w
              
            self.w=self.w-learning_rate*dw
            self.b=self.b-learning_rate*db
            
            #loss calculation
            """margin = y * (np.dot(X, self.w) + self.b)
            if j % 100 == 0:
                loss = C*np.sum(np.maximum(0, 1 - margin)) +  np.dot(self.w,self.w)
                epoch.append(j)
                loss_history.append(loss/len(X))"""
           
      
    def predict(self, X) -> np.ndarray:
        scr=np.dot(X,self.w)+self.b

        # make predictions for the given data
        return scr

    def accuracy_score(self, X, y) -> float:
        # compute the accuracy of the model (for debugging purposes)
        return np.mean(self.predict(X) == y)
    

class MultiClassSVM:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.models = []
        for i in range(self.num_classes):
            self.models.append(SupportVectorModel())

    
    def fit(self, X, y, **kwargs) -> None:
        # first preprocess the data to make it suitable for the 1-vs-rest SVM model
        # then train the 10 SVM models using the preprocessed data for each class  
        for i in range(self.num_classes):
            y_new=np.where(y==i,1,-1)
            self.models[i].fit(X,y_new,**kwargs)


    def predict(self, X) -> np.ndarray:
        # pass the data through all the 10 SVM models and return the class with the highest score
        scores = np.zeros((X.shape[0],self.num_classes))
        
        for i in range(self.num_classes):
            scores[:, i] = self.models[i].predict(X)
        
        
        
         
        return np.argmax(scores,axis=1) 

        
    def accuracy_score(self, X, y) -> float:
        return np.mean(self.predict(X) == y)
    
    def precision_score(self, X, y) -> float:
        precision_scores = []
        y_pred = self.predict(X)
        for i in range(self.num_classes):
          tp = np.sum((y == i) & (y_pred == i))
          fp = np.sum((y != i) & (y_pred == i))
          precision = tp / (tp + fp)
          precision_scores.append(precision)
        return np.mean(precision_scores)
        
         
    
    def recall_score(self, X, y) -> float:
        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred)
        
        # calculate recall for each class and average over all classes
        recalls = []
        for i in range(self.num_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0
            recalls.append(recall)
        
        return np.mean(recalls)
        
         
    
    def f1_score(self, X, y) -> float:
        y_pred = self.predict(X)
        
        # calculate f1 score using the confusion matrix
        precision = self.precision_score(X,y)
        recall = self.recall_score(X,y)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1