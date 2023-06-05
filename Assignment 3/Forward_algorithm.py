import numpy as np
from scipy.stats import norm




def forward(q, A, pX, finite):
    """
    Calculates the forward probabilities for a given observation sequence.

    Parameters:
    obs: array-like, shape (T_obs,)
        Observation sequence.
    q: array-like, shape (N_states,)
        Initial state distribution.
    A: array-like,
        shape (N_states, N_states) for infinite
        shape (N_states, N_states + 1) for finite
        Transition matrix.
    B: array-like, shape (N_states, T_obs)
        Observation matrix.
    finite: boolean
        finite or infinite duration HMM

    Returns:
    alpha: array-like, shape (T_obs, N_states)
        Forward variable.
    """
    T_obs = len(pX[0])
    N_states = len(q)

    alpha_temp = np.zeros((N_states, T_obs))
    c = np.zeros(T_obs)
    alpha = np.zeros((N_states, T_obs))

    # Initialization
    #alpha_temp[:, 0] = q * np.array([dist.pdf(obs[0]) for dist in B])
    alpha_temp[:, 0] = q * pX[:, 0]
    c[0] = sum(alpha_temp[:, 0])
    alpha[:, 0] = alpha_temp[:, 0]/(c[0])
    print("initial\n",alpha,"\n", c)

    # Recursion
    for t in range(1, T_obs):
        #alpha_temp[:, t] = np.array([dist.pdf(obs[t]) for dist in B]) * (alpha[:, t - 1].dot(A)[:-1])
        alpha_temp[:, t] = pX[:, t] * (alpha[:, t - 1].T.dot(A)[:-1])
        print("alpha_temp:\n", alpha_temp) 
        c[t] = sum(alpha_temp[:, t])
        alpha[:, t] = alpha_temp[:, t]/(c[t])
        print("iter",t,"\n", alpha, "\n", c)

    #termination
    if finite==True:
        c = np.append(c, (alpha[:, t].T.dot(A[:,-1])))
    print(c)
    return alpha, c


def logprob(obs, q, A, B, finite):
    alpha, c = forward(obs, q, A, B, finite)
    return sum(np.log(c))



#test code
q = np.array([1, 0])
A = np.array([[0.9, 0.1, 0], [0, 0.9, 0.1]])
B = [norm(0, 1), norm(3, 2)]
obs = [-0.2, 2.6, 1.3]

pX = np.zeros((2, len(obs)))
#normalize
for m in range(len(obs)):
    scalar = np.max(np.array([norm.pdf(obs[m], 0, 1), norm.pdf(obs[m], 3, 2)]))
    pX[0, m] = norm.pdf(obs[m], 0, 1) / scalar
    pX[1, m] = norm.pdf(obs[m], 3, 2) / scalar

print(pX)
alpha, c = forward(q, A, pX, True)
print("alpha_hat:\n", alpha)
print("c:\n", c)

'''
alpha_inf, c_inf = forward(obs, q, A[:, :-1], B, False)
print("alpha_hat inf:\n", alpha_inf)
print("c: inf\n", c_inf)
'''