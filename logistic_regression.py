import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd


def Sigmoid(x):
    return 1.0/(1+np.exp(-x))


def comLine(X,thetha):

    l=(X*thetha[1][0])+thetha[0][0]
    
    return l

def costFunction(X,y,thetha):

    m=len(y)
    
    a=-(y*np.log(Sigmoid(comLine(X,thetha))))
    b=-(1-y)*np.log(1-(Sigmoid(comLine(X,thetha))))
    print a
    print b
    res=a+b
    res=float(sum(res)/m)
    return res


def gradientDecent(X,y,thetha,alpha=0.01,iterations=1000):

    temp1=thetha[0][0]
    temp2=thetha[1][0]

    for i in range(iterations):
        a=float(alpha*sum(Sigmoid(comLine(X,[[temp1],[temp2]]))-y))
        b=float(alpha*sum((Sigmoid(comLine(X,[[temp1],[temp2]]))-y)*X))

        a=float(temp1-a)
        b=float(temp2-b)

        temp1=a
        temp2=b

    
    return np.array([[a],[b]],np.float64)


def regGraph(X,y,thetha):

    plt.scatter(X,y,label='scatter plot',c='#ef5423')
    
    plt.legend()
    plt.title('Logistic regression')
    plt.show()


def predict(p,thetha):
    t=comLine(p,thetha)
    print t
    x=Sigmoid(t)
    print x

    for i in range(len(x)):
        if x[i] >= 0.5:
            print 'there is {} percent chance that the poison no. {} is of the type 1'.format(x[i],i)

        else:
            print 'there is {} percent chance that the poison no.{} is of the type 0'.format((1-x[i]),i)
    


def main():
    X=np.ndarray([],np.float64)
    y=np.ndarray([],np.float64)
    thetha=np.array([[0],[0]],np.float64)
            
    df=pd.read_csv('poisons.csv')
    X=df.time
    y=df.poison

    print 'training............'
    thetha=gradientDecent(X,y,thetha)        
   

    print 'thetha0 : {} \nthetha1 : {}'.format(thetha[0][0],thetha[1][0])
    
    c=costFunction(X,y,thetha)
    print 'Error : {}',str(c)
    
    regGraph(X,y,thetha)

    predict(np.array([0.36,0.78]),thetha)

   
    

if __name__=='__main__':
    main()

    
    
    
