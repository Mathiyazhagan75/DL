import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# df = pd.DataFrame(data=[[0,0,0],[0,1,1],[1,0,1],[1,1,1]], columns=['x1','x2','y'])
df = pd.DataFrame(data=[[0,0,0],[0,1,0],[1,0,0],[1,1,1]], columns=['x1','x2','y'])
df
df.loc[:,'x0'] = 1
X = np.array(df[['x0','x1','x2']])
Y = np.array(df['y'])

def plot_graph(W):
    w0, w1, w2 = W
    x = df[['x0','x1','x2']]
    y = Y
    x1 = np.linspace(-2, 2, 100)
    x2 = (-w0-w1*x1)/w2

    plt.scatter(x['x1'], x['x2'], c=y, cmap='coolwarm')
    plt.plot(x1, x2, '-r')
    plt.legend(['Datapoints', 'Hyperplane - (wTx = 0)'])
    plt.show()

w_old = np.zeros(3)
w_new = np.zeros(3)
eta = 0.5
iterations =0
while(1):
    iterations+=1
    w_new = w_old.copy()
    for i in range(0,len(X)):
        yhat = 1 if np.dot(X[i],w_old) > 0 else 0
        if yhat == Y[i]:
            continue
        elif yhat==1 and Y[i]==0:
            w_old = w_old - eta * X[i]
        elif yhat==0 and  Y[i]==1:
            w_old = w_old + eta * X[i]
    plot_graph(w_old)
    if np.allclose(w_new,w_old):
        print(iterations)
        break

w_new
df = pd.DataFrame({'x1':[4, 4.5, 5, 5.5, 6, 6.5, 7.0], 'y':[33, 42, 45, 51, 53, 61, 62]})
df['x0']=1
X = np.array(df[['x0','x1']])
Y = np.array(df['y'])
w_old = np.zeros(2)
eta = 0.001
iteration=0
while(1):
    iteration+=1
    w_new = w_old.copy()
    total_grad = 0
    print("iteration ",iteration)
    print(w_new)
    for i in range(len(X)):
        yhat = np.dot(X[i],w_old)
        gradient = (Y[i]-yhat) * X[i]
        total_grad+=gradient
    w_old = w_old + eta* total_grad
    if np.allclose(w_old, w_new):
        break
