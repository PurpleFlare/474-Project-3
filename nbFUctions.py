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
        
        oneValues = np.count_nonzero(y == 1)
        twoValues = np.count_nonzero(y == 2)
    
        theta1 = (oneValue + a) / (y.size + a + b)
        theta2 = 1 - theta1
        
        thetaTuple = (theta1, theta2)
j = 1
        list_att = []
        list_prob = {}
        list_Y1 = {}
        list_Y2 = {}
        for j in range(X.shape[1]):
            list_Y1[j] = {}
            list_Y2[j] = {}
            #list_att.append(Counter(X[j]))
            myList = Counter(X[j]).keys()
            otherList = {}
            print(myList)
            print(Counter(X[j]).values())
            print(Counter(X[j]))
            kJ = np.unique(X[j]).size
            print(kJ)
            for k in range (1, kJ):
                prob = 0
                prob2 = 0
                s = 0
                #print(k)
                for l in range (X.shape[0]):
                    if k == X[l][j] and y[l] == 1:
                        s = s + 1
                    #elif myList[k] == X[l][j]      
                otherList = Counter(X[j])
                #print(otherList)
                prob = s / otherList[k] #Nj
                prob2 = 1 - prob
                #temp = (prob, prob2)
                #list_N[otherList[k]] = temp
                ans1 = (prob + alpha) / (oneValues + kJ*alpha)
                ans2 = (prob2 + alpha) / (twoValues + kJ*alpha)
                temp1 = (ans1, ans2)
                #list_prob[otherList[k]] = temp1
                list_Y1[j][k] = ans1
                list_Y2[j][k] = ans2
        
        #params = (thetaTuple, list_prob)
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
        for j in (Xtest.shape[0]):
            y1 = params["y=1"]
            y2 = params["y=2"]
            
            #find given product of Xj when y = 1 and y= 2
            temp = np.unique(Xtest[j])
            k = 1
            hgn = temp[0]
            productYes = params["X= " + hgn + " | y = 1"]
            productNo = params["X= " + hgn + " | y = 2"]
            for k in range (temp):
                var = temp[k]
                productYes = productYes * params["X= " + hgn + " | y = 1"]
                productNo = prodcuctNo * params["X= " + hgn + " | y = 2"]
            
            forYes = (y1 * productYes) / ((y1 * productYes) + (y2 * productNo))
            forNo = (y2 * productNo) / ((y2 *productNo) + (y1 * productYes))
            
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
    #remove next line and implement from here
    i = 0
    #getting value with bad credit
    y2 = 0
    for i in range (y_pred):
        if y_pred.at(i) == 2:
            y2 += 1
    #get unprivileged value from y_sensitive
    j = 0
    sucks2Bme = 0
    imOKtho = 0
    for j in (y_sensitive):
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
