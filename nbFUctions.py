from sklearn.base import BaseEstimator
import numpy as np
import scipy.stats as stats
from collections import Counter
# For this assignment we will implement the Naive Bayes classifier as a
# a class, sklearn style. You only need to modify the fit and predict functions.
# Additionally, implement the Disparate Impact measure as the evaluateBias function.
class NBC(BaseEstimator):
    '''
    (a,b) - Beta prior parameters for the class random variable
    alpha - Symmetric Dirichlet parameter for the features
    '''

    def __init__(self, a=1, b=1, alpha=1):
        self.a = a
        self.b = b
        self.alpha = alpha
        self.__params = None
        
    def get_a(self):
        return self.a

    def get_b(self):
        return self.b

    def get_alpha(self):
        return self.alpha

    # you need to implement this function

    def fit(self,X,y):
        '''
        This function does not return anything
        
        Inputs:
        X: Training data set (N x d numpy array)
        y: Labels (N length numpy array)
        '''
        a = self.get_a()
        b = self.get_b()
        alpha = self.get_alpha()
        self.__classes = np.unique(y)
        params = {}
        # remove next line and implement from here
        # you are free to use any data structure for paramse
        
        oneValues = np.count_nonzero(y == 1)
        twoValues = np.count_nonzero(y == 2)
    
        theta1 = (oneValues + a) / (y.size + a + b)
        theta2 = 1 - theta1
        
        thetaTuple = (theta1, theta2)
        list_Y1 = {}
        list_Y2 = {}
      
        for j in range(X.shape[1]):
            list_Y1[j] = {}
            list_Y2[j] = {}
           
            kJ = np.unique(X[:,j]).shape[0]
            mJ = np.max(X[:,j])
            
            for k in range (0, mJ+1):
                s = 0
                s2 = 0
                
                for l in range (X.shape[0]):
                    if k == X[l][j] and y[l] == 1:
                        s = s + 1
                for l in range (X.shape[0]):
                    if k == X[l][j] and y[l] == 2:
                        s2 = s2 + 1 
                
                ans1 = (s + alpha) / (oneValues + kJ*alpha)
                
                ans2 = (s2 + alpha) / (twoValues + kJ*alpha)
                
                list_Y1[j][k] = ans1
                list_Y2[j][k] = ans2
        
        
            list_Y1[j][-1] = (alpha) / (oneValues + kJ*alpha)
            list_Y2[j][-1] = (alpha) / (twoValues + kJ*alpha)
            
        params = (thetaTuple, list_Y1, list_Y2)
        
       
        # do not change the line below
        self.__params = params
    
    # you need to implement this function
    def predict(self,Xtest):
        '''
        This function returns the predicted class for a given data set
        
        Inputs:
        Xtest: Testing data set (N x d numpy array)
        
        Output:
        predictions: N length numpy array containing the predictions
        '''
        params = self.__params
        a = self.get_a()
        b = self.get_b()
        alpha = self.get_alpha()
        #remove next line and implement from here
        j = 0
        predictions = []
        for j in range (Xtest.shape[0]):
            y1 = params[0][0]
            y2 = params[0][1]
            
            #find given product of Xj when y = 1 and y= 2
            prod1 = 1
            prod2 = 1
            for k in range (Xtest.shape[1]):
                temp = Xtest[j][k] 
                
                #when attribute k has y = 1
                #when attribute k has y = 1
                #1 means the first library
                #k means each attribute
                #temp is the feature of kth attribute that jth person has
                if(temp in params[1][k].keys()):
                    val1 = params[1][k][temp]
                else:
                    val1 = params[1][k][-1]
                if(temp in params[2][k].keys()):
                #when attribute k has y = 2
                    val2 = params[2][k][temp]
                else:
                    val2 = params[2][k][-1]
                prod1 = prod1 * val1
                prod2 = prod2 * val2
            
            forYes = (y1 * prod1) / ((y1 * prod1) + (y2 * prod2))
            forNo = (y2 * prod2) / ((y2 *prod2) + (y1 * prod1))
            
            if forYes > forNo:
                predictions.append(1)
            else:
                predictions.append(2)
        #do not change the line below
        return predictions
        
def evaluateBias(y_pred,y_sensitive):
    '''
    This function computes the Disparate Impact in the classification predictions (y_pred),
    with respect to a sensitive feature (y_sensitive).
    
    Inputs:
    y_pred: N length numpy array
    y_sensitive: N length numpy array
    
    Output:
    di (disparateimpact): scalar value
    '''
    arrayS = np.count_nonzero(y_sensitive == 1)
    arrayNS = np.count_nonzero(y_sensitive == 2)
    
  
    numeratorVal = 0
    denominatorVal = 0
    
    for i in range(y_pred.size):
        if (y_pred[i] == 2 and y_sensitive[i] != 1):
            numeratorVal += 1
   
        elif (y_pred[i] == 2 and y_sensitive[i] == 1):
            denominatorVal += 1
            
            
    numeratorVal = numeratorVal/arrayNS
    denominatorVal = denominatorVal/arrayS
    
    di = numeratorVal/denominatorVal

    return di
def genBiasedSample(X,y,s,p,nsamples=1000):
    '''
    Oversamples instances belonging to the sensitive feature value (s != 1)
    
    Inputs:
    X - Data
    y - labels
    s - sensitive attribute
    p - probability of sampling unprivileged customer
    nsamples - size of the resulting data set (2*nsamples)
    
    Output:
    X_sample,y_sample,s_sample
    '''
    i1 = y == 1 # good
    i1 = i1[:,np.newaxis]
    i2 = y == 2 # bad
    i2 = i2[:,np.newaxis]
    
    sp = s == 1 #privileged
    sp = sp[:,np.newaxis]
    su = s != 1 #unprivileged
    su = su[:,np.newaxis]
    su1 = np.where(np.all(np.hstack([su,i1]),axis=1))[0]
    su2 = np.where(np.all(np.hstack([su,i2]),axis=1))[0]
    sp1 = np.where(np.all(np.hstack([sp,i1]),axis=1))[0]
    sp2 = np.where(np.all(np.hstack([sp,i2]),axis=1))[0]
    inds = []
    for i in range(nsamples):
        u = stats.bernoulli(p).rvs(1)
        if u == 1:
            #sample one bad instance with s != 1
            inds.append(np.random.choice(su2,1)[0])
            #sample one good instance with s == 1
            inds.append(np.random.choice(sp1,1)[0])
        else:
            #sample one good instance with s != 1
            inds.append(np.random.choice(su1,1)[0])
            #sample one bad instance with s == 1
            inds.append(np.random.choice(sp2,1)[0])
    X_sample = X[inds,:]
    y_sample = y[inds]
    s_sample = s[inds]
    
    return X_sample,y_sample,s_sample,inds
