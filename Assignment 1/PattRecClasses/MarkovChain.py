import numpy as np
#from .DiscreteD import DiscreteD
from DiscreteD import DiscreteD
from scipy.stats import norm

class MarkovChain:
    """
    MarkovChain - class for first-order discrete Markov chain,
    representing discrete random sequence of integer "state" numbers.
    
    A Markov state sequence S(t), t=1..T
    is determined by fixed initial probabilities P[S(1)=j], and
    fixed transition probabilities P[S(t) | S(t-1)]
    
    A Markov chain with FINITE duration has a special END state,
    coded as nStates+1.
    The sequence generation stops at S(T), if S(T+1)=(nStates+1)
    """
    def __init__(self, initial_prob, transition_prob):


        if not (np.all(initial_prob >= 0) and np.isclose(np.sum(initial_prob), 1)):
            raise ValueError("Initial probabilities must be non-negative and sum up to 1.")
        if not (np.all(transition_prob >= 0) and np.isclose(np.sum(transition_prob, axis=1), 1)).all():
            raise ValueError("Transition probabilities must be non-negative and row-wise sum up to 1.")
        
        self.q = initial_prob    #InitialProb(i)= P[S(1) = i]
        self.A = transition_prob #TransitionProb(i,j)= P[S(t)=j | S(t-1)=i]
        self.nStates = transition_prob.shape[0]
        self.is_finite = False

        if self.A.shape[0] != self.A.shape[1]:
            self.is_finite = True


    def probDuration(self, tmax):
        """
        Probability mass of durations t=1...tMax, for a Markov Chain.
        Meaningful result only for finite-duration Markov Chain,
        as pD(:)== 0 for infinite-duration Markov Chain.
        
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.8.
        """
        if not isinstance(tmax, int) or tmax <= 0:
            raise ValueError("tmax must be a positive integer.")

        pD = np.zeros(tmax)

        if self.is_finite:
            pSt = (np.eye(self.nStates)-self.A.T)@self.q

            for t in range(tmax):
                pD[t] = np.sum(pSt)
                pSt = self.A.T@pSt

        return pD

    def probStateDuration(self, tmax):
        """
        Probability mass of state durations P[D=t], for t=1...tMax
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.7.
        """
        if not isinstance(tmax, int) or tmax <= 0:
            raise ValueError("tmax must be a positive integer.")
                
        t = np.arange(tmax).reshape(1, -1)
        aii = np.diag(self.A).reshape(-1, 1)
        
        logpD = np.log(aii)*t+ np.log(1-aii)
        pD = np.exp(logpD)

        return pD

    def meanStateDuration(self):
        """
        Expected value of number of time samples spent in each state
        """
        return 1/(1-np.diag(self.A))
    
    def rand(self, tmax):
        """
        S=rand(self, tmax) returns a random state sequence from given MarkovChain object.
        
        Input:
        tmax= scalar defining maximum length of desired state sequence.
           An infinite-duration MarkovChain always generates sequence of length=tmax
           A finite-duration MarkovChain may return shorter sequence,
           if END state was reached before tmax samples.
        
        Result:
        S= integer row vector with random state sequence,
           NOT INCLUDING the END state,
           even if encountered within tmax samples
        If mc has INFINITE duration,
           length(S) == tmax
        If mc has FINITE duration,
           length(S) <= tmaxs
        """
        if not isinstance(tmax, int) or tmax <= 0:
            raise ValueError("tmax must be a positive integer.")

        # Initialize variables
        S = np.array([])
        i = DiscreteD(self.q).rand(1)
        duration = 0
        
        # Generate state sequence
        while duration < tmax:
            S = np.concatenate((S,i))
            duration += 1
            if i == self.nStates and self.is_finite :    # END state
                break

            p = self.A[i, :][0]
            j = DiscreteD(p).rand(1)
            i = j
    
        return S.astype(int)
    


    def viterbi(self):
        pass
    
    def stationaryProb(self):
        pass
    
    def stateEntropyRate(self):
        pass
    
    def setStationary(self):
        pass



    def join(self):
        pass

    def initLeftRight(self):
        pass
    
    def initErgodic(self):
        pass

    def forward(self, pX):
        """
        Calculates the forward probabilities for a given observation sequence.

        Parameters:
        pX: array-like
            The observation sequence (scaled).

        Returns:
        alpha: array-like, shape (T_obs, N_states)
            Forward variable.
        """
        T_obs = len(pX[0])
        N_states = len(self.q)

        alpha_temp = np.zeros((N_states, T_obs))
        c = np.zeros(T_obs)
        alpha = np.zeros((N_states, T_obs))

        # Initialization
        #alpha_temp[:, 0] = q * np.array([dist.pdf(obs[0]) for dist in B])
        alpha_temp[:, 0] = self.q * pX[:, 0]
        c[0] = sum(alpha_temp[:, 0])
        alpha[:, 0] = alpha_temp[:, 0]/(c[0])

        # Recursion
        for t in range(1, T_obs):
            #alpha_temp[:, t] = np.array([dist.pdf(obs[t]) for dist in B]) * (alpha[:, t - 1].dot(A)[:-1])
            alpha_temp[:, t] = pX[:, t] * (alpha[:, t - 1].T.dot(self.A)[:-1])
            c[t] = sum(alpha_temp[:, t])
            alpha[:, t] = alpha_temp[:, t]/(c[t])

        #termination
        if self.is_finite==True:
            c = np.append(c, (alpha[:, t].T.dot(self.A[:,-1])))
        return alpha, c
    


    def finiteDuration(self):
        pass
    
    def backward(self, pX):
        """
        Calculates the backward probabilities for a given observation sequence.

        Parameters:
        obs: array-like
            The observation sequence.

        Returns:
        beta: array-like, shape (T_obs, N_states)
            Backward variable.
        """
        T_obs = len(pX[0])
        N_states = len(self.q)

        beta = np.zeros((N_states, T_obs))
        beta_hat = np.zeros((N_states, T_obs))
        
        _,c = self.forward(pX)

        print(c)
        print(pX)
        # Initialization
        if self.is_finite:
            beta[:, T_obs - 1] = self.A[:, N_states]
            beta_hat[:, T_obs - 1] = beta[:, T_obs - 1]/(c[T_obs - 1]*c[T_obs])
        else:

            beta[:, T_obs - 1] = 1
            beta_hat[:, T_obs - 1] = 1/c[T_obs - 1]

        # Recursion
        for t in range(T_obs - 2, -1, -1):
            if self.is_finite:
                beta_hat[:, t] = (self.A[:, :-1].dot(pX[:, t + 1]*beta_hat[:, t+1]))/c[t]
            else:
                beta_hat[:, t] = (self.A.dot(pX[:, t + 1]*beta_hat[:, t+1]))/c[t]

        return beta_hat
    

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass