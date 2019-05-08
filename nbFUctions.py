from sklearn.base import BaseEstimator
import numpy as np
import scipy.stats as stats

# For this assignment we will implement the Naive Bayes classifier as a
# a class, sklearn style. You only need to modify the fit and predict functions.
# Additionally, implement the Disparate Impact measure as the evaluateBias function.
class NBC(BaseEstimator):
    '''
    (a,b) - Beta prior parameters for the class random variable
    alpha - Symmetric Dirichlet parameter for the features
    '''

    def __init__(self, a=1, b=1, alpha=1):
        self.__a = a
        self.__b = b
        self.__alpha = alpha
        self.__params = None
        
    def get_a(self):
        return self.__a

    def get_b(self):
        return self.__b

    def get_alpha(self):
        return self.__alpha

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
        #i = 0
        #yes = 0
        #no = 0
        #for i in range y.shape[0]:
        #    if y.at(i) == 1 :
        #        yes = yes +1
        #    else:
        #        no = no + 1
        #bayes1 = (yes + a) / (y.size + a + b)
        #bayes2 = 1 - bayes1
        #params["y=1"] = bayes1
        #params["y=2"] = bayes2
        
        for i in range y.size-1:
        oneValues = np.where(y[i]==1)
        twoValues = np.where(y[i]==2)
        
        theta1 = (oneValues + a) / (y.size + a + b)
        theta2 = 1 - theta1
        
        params["y=1"] = theta1
        params["y=2"] = theta2

        
        
        for j in range X.shape[1]:
            #the Kj value
            temp = np.unique(X.at(j))
            for k in range temp.shape[0]
                temp1 = np.where(X[j] == temp[k])
                temp2 = np.where(y == 1) 
                temp4 = np.where(y == 2)
                temp3 = np.intersect(temp1, temp2)
                temp5 = np.intersect(temp1, temp4)
                
                prob1 = temp3.size / temp1.size
                prob2 = temp5.size / temp1.size
                
                total1 = (prob1 + alpha)  / (oneValues + temp.size*alpha)
                total2 = (prob2 + alpha) / (twoValues + temp.size*alpha)
                
                params["X =" + j + " | y = 1"] = total1    
                params["X =" + j + " | y = 2"] = total2
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
    #remove next line and implement from here
    int i = 0
    #getting value with bad credit
    int y2
    for i in y_pred:
        if y_pred.at(i) == 2:
            y2 += 1
    #get unprivileged value from y_sensitive
    int j = 0
    int sucks2Bme
    int imOKtho
    for j in y_sensitive:
        if y_sensitive.at(i) == 2:
            sucks2Bme += 1
        else:
            imOKtho += 1
    
    di = (y2 * imOKtho) / (y2 * sucks2Bme)
    
    #do not change the line below
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
