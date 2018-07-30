import numpy as np 
import pandas as pd

def cost_function(ans ,y):
    print(np.sum((ans-y)*(ans-y)))
    return(np.sum((ans-y)*(ans-y)))

def sigmoid(z):
    return(1/(1+np.exp(z)))



def initialize_theta(theta,tsize,a,xn,yn):
    for i in range(tsize):
        if(i==0):
            theta.append(np.random.randn(xn+1,a[0]))
        elif(i==tsize-1):
            theta.append(np.random.randn(a[i-1]+1,yn))
        else:
            theta.append(np.random.randn(a[i-1]+1,a[i]))
        
    return(theta)
            
def making_hidden(m,a):
    hidden = []
    for i in a :
        hidden.append(np.ones((m,i+1)))
    return(hidden)

def forward(x,theta,hidden):
    
    
    hidden[0][:,1:]= sigmoid(np.dot(x,theta[0]))
    for i in range(1,len(hidden)-1):
        
        hidden[i][:,1:]= sigmoid(np.dot(hidden[i-1],theta[i]))
    ans = sigmoid(np.dot(hidden[-1],theta[-1]))
    return(hidden,ans)
        
def deltas(y , ans , theta,hidden, x):
    delta = []
    delta.append(y-ans)
    delta.insert(0,((np.dot(delta[0],theta[len(hidden)].T))*(hidden[-1]*(1-hidden[-1]))))
    for i in range(len(hidden)-2, -1 , -1):
        delta.insert(0,((np.dot(delta[0][:,1:],theta[i+1].T))*(hidden[i]*(1-hidden[i]))))
    return(delta)
def gradients(delta,hidden,x):
    gradient = []
    gradient.append(np.dot(hidden[-1].T,delta[-1]))
    for i in range(len(hidden)-2,-1,-1):
        k= delta[i+1][:,1:]
        gradient.insert(0,np.dot(hidden[i].T,k))
    gradient.insert(0,(np.dot(x.T,delta[0][:,1:])))
    return(gradient)
def update(x,y,m,theta,hidden,iterations, alpha):
    hidden , ans = forward(x,theta,hidden)
    b = cost_function(ans,y)
    for i in range(iterations):
        hidden , ans = forward(x,theta,hidden)
        delta = deltas(y,ans,theta,hidden,x)
        gradient = gradients(delta , hidden , x)
        if(b<cost_function(ans,y)):
            alpha = alpha/2
            
        for j in range(len(theta)):  
            theta[j]= theta[j]-alpha*gradient[j]
        b = cost_function(ans,y)
    return(theta)
        
def main():
    data = np.loadtxt('Corners.txt')
    
    x1 = data[:,:2]
    y1 = data[:,2]
    y = np.zeros((np.size(y1),4))
    for i in range(np.size(y1)):
        y[i,int(y1[i])]=1
    print(y)
        
    
    
    m = np.size(x1,0) 
    xn = np.size(x1,1)
    yn = np.size(y,1)
    x = np.ones((m,xn+1))
    x[:,1:]= x1
   
    a= [2]
    
    
    theta =[]
    tsize = len(a)+1
    theta = initialize_theta(theta,tsize,a,xn,yn)  
    hidden = making_hidden(m,a)
    
    iterations = 100
    alpha = 0.3
    theta = update(x,y,m,theta,hidden,iterations,alpha)
    
    
    
    
    

    
    
    hidden , ans = forward(x, theta, hidden)
    print(np.round(ans))
    q = 0 
    q1 = 0 
    for i in range(np.size(ans,0)):
        t = 1
        for j in range(4):
            if(np.round(ans[i][j])!=y[i][j]):
                t = 0
                 
        if(t==1):
            q+=1
        q1+=1
    print(ans[0][0])
    
    
    
    
main()