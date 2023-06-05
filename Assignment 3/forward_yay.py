import numpy as np
from scipy.stats import norm

def forward(x):
    '''
    Input:
    x: observed sequence 1*t
    Output:
    alpha_hat: scaled forward variable 2*t
    c: forward scale factors. Infinite: 1*t, finite: 1*(t+1)
    '''
    #Hidden Markov Model
    q = np.array([[1], [0]])
    A = np.array([[0.9, 0.1, 0], [0, 0.9, 0.1]])
    mu1 = 0
    sigma1 = 1
    mu2 = 3
    sigma2 = 2
    pX = np.zeros((2, len(x)))
    #normalize
    for m in range(len(x)):
        print(norm.pdf(x[m], mu1, sigma1))
        print(norm.pdf(x[m], mu2, sigma2))
        print(np.max(np.array([norm.pdf(x[m], mu1, sigma1), norm.pdf(x[m], mu2, sigma2)])))
        scalar = np.max(np.array([norm.pdf(x[m], mu1, sigma1), norm.pdf(x[m], mu2, sigma2)]))
        pX[0, m] = norm.pdf(x[m], mu1, sigma1) / scalar
        pX[1, m] = norm.pdf(x[m], mu2, sigma2) / scalar

    print("pX:\n", pX)
    # Initialize
    alpha_hat =  np.zeros((2, len(x)))
    c = []
    alpha_tem = np.zeros((len(q), 1))
    alpha_tem[0, 0] = q[0, 0] * pX[0,0]
    alpha_tem[1, 0] = q[1, 0] * pX[1,0]
    print(norm.pdf(x[0], mu1, sigma2))
    c.append(sum(alpha_tem))
    alpha_hat[:, 0] = (alpha_tem[:, 0]/c[0]).T

    #Forward step
    for t in range(1,len(x)):
        alpha_a_tem = np.matmul(alpha_hat[:, t - 1].T, A)
        alpha_tem[0, 0] = pX[0,t]*alpha_a_tem[0]
        alpha_tem[1, 0] = pX[1,t]*alpha_a_tem[1]
        c.append(sum(alpha_tem))
        alpha_hat[:, t] = (alpha_tem/c[t]).T
        print("iter",t,"\n")
        print(alpha_tem)
        print(alpha_hat)

    #Termination
    c.append(np.matmul(alpha_hat[:, t].T, A[:,-1]))
    return alpha_hat, c

obs = [-0.2, 2.6, 1.3]

alpha, c = forward(obs)
print("alpha_hat:\n", alpha)
print("c:\n", c)