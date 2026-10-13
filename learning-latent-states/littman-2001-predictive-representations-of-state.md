---
title: "Predictive Representations of State"
authors: ["Michael L. Littman", "Richard S. Sutton", "Satinder Singh"]
year: 2001
url: "https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf"
source: "NeurIPS 2001 PDF, converted with pymupdf4llm"
---

# Predictive Representations of State 

Michael L. Littman 

Richard S. Sutton AT&T Labs-Research, Florham Park, New Jersey 

{mlittman,sutton}~research.att.com 

Satinder Singh 

Syntek Capital, New York, New York 

baveja~cs.colorado.edu 

## Abstract 

We show that states of a dynamical system can be usefully represented by multi-step, action-conditional predictions of future observations. State representations that are grounded in data in this way may be easier to learn, generalize better, and be less dependent on accurate prior models than, for example, POMDP state representations. Building on prior work by Jaeger and by Rivest and Schapire, in this paper we compare and contrast a linear specialization of the predictive approach with the state representations used in POMDPs and in k-order Markov models. Ours is the first specific formulation of the predictive idea that includes both stochasticity and actions (controls). We show that any system has a linear predictive state representation with number of predictions no greater than the number of states in its minimal POMDP model. 

In predicting or controlling a sequence of observations, the concepts of state and state estimation inevitably arise. There have been two dominant approaches. The _generative-model_ approach, typified by research on partially observable Markov decision processes (POMDPs), hypothesizes a structure for generating observations and estimates its state and state dynamics. The _history-based_ approach, typified by k-order Markov methods, uses simple functions of past observations as state, that is, as the immediate basis for prediction and control. (The data flow in these two approaches are diagrammed in Figure 1.) Of the two, the generative-model approach is more general. The model's internal state gives it temporally unlimited memorythe ability to remember an event that happened arbitrarily long ago--whereas a history-based approach can only remember as far back as its history extends. The bane of generative-model approaches is that they are often strongly dependent on a good model of the system's dynamics. Most uses of POMDPs, for example, assume a perfect dynamics model and attempt only to estimate state. There are algorithms for simultaneously estimating state and dynamics (e.g., Chrisman, 1992), analogous to the Baum-Welch algorithm for the uncontrolled case (Baum et al., 1970), but these are only effective at tuning parameters that are already approximately correct (e.g., Shatkay & Kaelbling, 1997). 



<!-- Start of picture text -->
observations¢E<br>observations state (and actions) / state<br>(and actions) 1-----1-----1..- rep'n 1-stept/' --+ . rep'n<br>delays<br>(a) (b)<br><!-- End of picture text -->

Figure 1: Data flow in a) POMDP and other recursive updating of state representation, and b) history-based state representation. 

In practice, history-based approaches are often much more effective. Here, the state representation is a relatively simple record of the stream of past actions and observations. It might record the occurrence of a specific subsequence or that one event has occurred more recently than another. Such representations are far more closely linked to the data than are POMDP representations. One way of saying this is that POMDP learning algorithms encounter many local minima and saddle points because all their states are equipotential. History-based systems immediately break symmetry, and their direct learning procedure makes them comparably simple. McCallum (1995) has shown in a number of examples that sophisticated history-based methods can be effective in large problems, and are often more practical than POMDP methods even in small ones. 

The _predictive state representation_ (PSR) approach, which we develop in this paper, is like the generative-model approach in that it updates the state representation recursively, as in Figure l(a), rather than directly computing it from data. We show that this enables it to attain generality and compactness at least equal to that of the generative-model approach. However, the PSR approach is also like the history-based approach in that its representations are grounded in data. Whereas a history-based representation looks to the past and records what did happen, a PSR looks to the future and represents what _will_ happen. In particular, a PSR is a vector of predictions for a specially selected set of action-observation sequences, called _tests_ (after Rivest & Schapire, 1994). For example, consider the test _U101U202,_ where _U1_ and _U2_ are specific actions and 01 and 02 are specific observations. The correct prediction for this test given the data stream up to time _k_ is the probability of its observations occurring (in order) given that its actions are taken (in order) (i.e., _Pr {Ok_ = _01, Ok+1_ = 02 I _A k_ = _u1,A k+ 1_ = _U2})._ Each test is a kind of experiment that could be performed to tell us something about the system. If we knew the outcome of all possible tests, then we would know everything there is to know about the system. A PSR is a set of tests that is sufficient information to determine the prediction for all possible tests (a sufficient statistic). 

As an example of these points, consider the float/reset problem (Figure 2) consisting of a linear string of 5 states with a distinguished _reset_ state on the far right. One action, f (float), causes the system to move uniformly at random to the right or left by one state, bounded at the two ends. The other action, r (reset), causes a jump to the reset state irrespective of the current state. The observation is always ounless the r action is taken when the system is already in the reset state, in which case the observation is 1. Thus, on an f action, the correct prediction is always 0, whereas on an r action, the correct prediction depends on how many fs there have been since the last r: for zero fS, it is 1; for one or two fS, it is 0.5; for three or four fS, it is 0.375; for five or six fs, it is 0.3125, and so on decreasing after every second f, asymptotically bottoming out at 0.2. 

No k-order Markov method can model this system exactly, because no limited-. 



<!-- Start of picture text -->
.5 .5<br>a) float action b) reset action<br><!-- End of picture text -->



<!-- Start of picture text -->
1,0=1<br><!-- End of picture text -->

Figure 2: Underlying dynamics of the float/reset problem for a) the float action and b) the reset action. The numbers on the arcs indicate transition probabilities. The observation is always 0 except on the reset action from the rightmost state, which produces an observation of 1. 

length history is a sufficient statistic. A POMDP approach can model it exactly by maintaining a belief-state representation over five or so states. A PSR, on the other hand, can exactly model the float/reset system using just two tests: rl and fOrI. Starting from the rightmost state, the correct predictions for these two tests are always two successive probabilities in the sequence given above (1, 0.5, 0.5, 0.375,...), which is always a sufficient statistic to predict the next pair in the sequence. Although this informational analysis indicates a solution is possible in principle, it would require a nonlinear updating process for the PSR. 

In this paper we restrict consideration to a linear special case of PSRs, for which we can guarantee that the number of tests needed does not exceed the number of states in the minimal POMDP representation (although we have not ruled out the possibility it can be considerably smaller). Of greater ultimate interest are the prospects for learning PSRs and their update functions, about which we can only speculate at this time. The difficulty of learning POMDP structures without good prior models are well known. To the extent that this difficulty is due to the indirect link between the POMDP states and the data, predictive representations may be able to do better. 

Jaeger (2000) introduced the idea of predictive representations as an alternative to belief states in hidden Markov models and provided a learning procedure for these models. We build on his work by treating the control case (with actions), which he did not significantly analyze. We have also been strongly influenced by the work of Rivest and Schapire (1994), who did consider tests including actions, but treated only the deterministic case, which is significantly different. They also explored construction and learning algorithms for discovering system structure. 

## 1 Predictive State Representations 

We consider dynamical systems that accept actions from a discrete set _A_ and generate observations from a discrete set _O._ We consider only predicting the system, not controlling it, so we do not designate an explicit reward observation. We refer to such a system as an _environment._ We use the term _history_ to denote a test forming an initial stream of experience and characterize an environment by a probability distribution over all possible histories, _P_ : _{OIA}*_ H- [0,1], where _P(Ol··· Otl a1··· at)_ is the probability of observations _01,_ ... , O£ being generated, in that order, given that actions _aI,_ ... _,at_ are taken, in that order. The probability of a test _t_ conditional on a history _h_ is defined as _P(tlh)_ = _P(ht)/P(h)._ Given a set of _q_ tests _Q_ = _{til,_ we define their (1 x _q) prediction vector, p(h)_ = _[P(t1Ih),P(t2Ih), ... ,P(tqlh)],_ as a predictive state representation (PSR) if and only if it forms a sufficient statistic for the environment, Le., if and only if 



for any test t and history _h,_ and for some _projection junction ft_ : [0, _l]q_ ~ [0,1]. In this paper we focus on _linear PSRs,_ for which the projection functions are linear, that is, for which there exist a (1 x _q) projection vector mt,_ for every test t, such that 



for all histories _h._ 

Let _Pi(h)_ denote the ith component of the prediction vector for some PSR. This can be updated recursively, given a new action-observation pair _a,o,_ by 



where the last step is specific to linear PSRs. We can now state our main result: 

Theorem 1 _For any environment that can be represented by a finite POMDP model, there exists a linear PSR with number of tests no larger than the number of states in the minimal POMDP model._ 

2 Proof of Theorem 1: Constructing a PSR from a POMDP 

We prove Theorem 1 by showing that for any POMDP model of the environment, we can construct in polynomial time a linear PSR for that POMDP of lesser or equal complexity that produces the same probability distribution over histories as the POMDP model. 

We proceed in three steps. First, we review POMDP models and how they assign probabilities to tests. Next, we define an algorithm that takes an n-state POMDP model and produces a set of _n_ or fewer tests, each of length less than or equal to _n._ Finally, we show that the set of tests constitute a PSR for the POMDP, that is, that there are projection vectors that, together with the tests' predictions, produce the same probability distribution over histories as the POMDP. 

A POMDP (Lovejoy, 1991; Kaelbling et al., 1998) is defined by a sextuple _(8, A,_ 0, _bo,T,_ 0). Here, 8 is a set of _n_ underlying (hidden) states, _A_ is a discrete set of actions, and 0 is a discrete set of observations. The (1 x _n)_ vector _bo_ is an initial state distribution. The set _T_ consists of _(n_ x _n)_ transition matrices _Ta,_ one for each action _a,_ where _Tlj_ is the probability of a transition from state i to _j_ when action _a_ is chosen. The set 0 consists of diagonal _(n_ x _n)_ observation matrices _oa,o,_ one for each pair of observation 0 and action _a,_ where o~'o is the probability of observation 0 when action _a_ is selected and state i is reached.<sup>l</sup> 

The state representation in a POMDP (Figure l(a)) is the _belief state-the_ (1 x _n)_ vector of the state-occupation probabilities given the history _h._ It can be computed recursively given a new action _a_ and observation 0 by 



where _en_ is the (1 x n)-vector of all Is. 

Finally, a POMDP defines a probability distribution over tests (and thus histories) by 



IThere are many equivalent formulations and the conversion procedure described here can be easily modified to accommodate other POMDP definitions. 

We now present our algorithm for constructing a PSR for a given POMDP. It uses a function _u_ mapping tests to (1 x _n)_ vectors defined recursively by u(c) == _en_ and _u(aot)_ == _(Taoa,ou(t)T)T,_ where c represents the null test. Conceptually, the components of _u(t)_ are the probabilities of the test _t_ when applied from each underlying state of the POMDP; we call _u(t)_ the _outcome vector_ for test _t._ We say a test _t_ is _linearly independent_ of a set of tests _S_ if its outcome vector is linearly independent of the set of outcome vectors of the tests in _S._ Our algorithm search is used and defined as 

_Q_ -<- search(c, {}) 

search(t, _S):_ for each _a_ E _A, 0_ E 0 if _aot_ is linearly independent of _S_ then _S_ -<- _search(aot, S_ U _{aot})_ return _S_ 

The algorithm maintains a set of tests and searches for new tests that are linearly independent of those already found. It is a form of depth-first search. The algorithm halts when it checks all the one-step extensions of its tests and finds none that are linearly independent. Because the set of tests _Q_ returned by search have linearly independent outcome vectors, the cardinality of _Q_ is bounded by _n,_ ensuring that the algorithm halts after a polynomial number of iterations. Because each test in _Q_ is formed by a one-step extension to some other test in _Q,_ no test is longer than _n_ action-observation pairs. 

The check for linear independence can be performed in many ways, including Gaussian elimination, implying that search terminates in polynomial time. 

By construction, all one-step extensions to the set of tests _Q_ returned by search are linearly dependent on those in _Q._ We now show that this is true for any test. 

Lemma 1 _The outcome vectors of the tests in Q can be linearly combined to produce the outcome vector for any test._ 

Proof: Let _U_ be the _(n_ x _q)_ matrix formed by concatenating the outcome vectors for all tests in _Q._ Since, for all combinations of _a_ and 0, the columns of _Taoa,ou_ are linearly dependent on the columns of _U,_ we can write _Taoa,ou_ == _UW_<sup>_T_</sup> for some _q_ x _q_ matrix of weights _W._ 

If t is a test that is linearly dependent on _Q,_ then anyone-step extension of _t, aot,_ is linearly dependent on _Q._ This is because we can write the outcome vector for _t_ as _u(t)_ == _(UwT)T_ for some (1 x _q)_ weight vector _w_ and the outcome vector for _aot_ as _u(aot)_ == _(Taoa,ou(t)T)T_ == _(Taoa,oUwT)T_ == _(UWTwT)T._ Thus, _aot_ is linearly dependent on _Q._ 

Now, note that all one-step tests are linearly dependent on _Q_ by the structure of the search algorithm. Using the previous paragraph as an inductive argument, this implies that _all_ tests are linearly dependent on _Q._ 0 

Returning to the float/reset example POMDP, search begins with by enumerating the 4 extensions to the null test (fO, fl, rO, and rl). Of these, only fa and rO are are linearly independent. Of the extensions of these, fOrO is the only one that is linearly independent of the other two. The remaining two tests added to _Q_ by search are fOfOrO and fOfOfOrO. No extensions of the 5 tests in _Q_ are linearly independent of the 5 tests in _Q,_ so the procedure halts. 

We now show that the set of tests Q constitute a PSR for the POMDP by constructing projection vectors that, together with the tests' predictions, produce the same probability distribution over histories as the POMDP. 

For each combination of _a_ and 0, define a _q_ x _q_ matrix _Mao_ == _(U+Taoa,ou)T_ and a 1 x _q_ vector _mao_ == _(U+Taoa,oe;;J_<sup>_T_</sup> _,_ where _U_ is the matrix of outcome vectors defined in the previous section and _U+_ is its pseudoinverse<sup>2</sup> • The ith row of _Mao_ is _maoti._ The probability distribution on histories implied by these projection vectors is 





Le., it is the same as that of the POMDP, as in Equation 4. Here, the last step uses the fact that _UU+v_<sup>_T_</sup> == _v_<sup>_T_</sup> for _v_<sup>_T_</sup> linearly dependent on the columns of _U._ This holds by construction of _U_ in the previous section. 

This completes the proof of Theorem 1. 

Completing the float/reset example, consider the Mf,o matrix found by the process defined in this section. It derives predictions for each test in Q after taking action f. Most of these are quite simple because the tests are so similar: the new prediction for rO is exactly the old prediction for fOrO, for example. The only non trivial test is fOfOfOrO. Its outcome can be computed from 0.250 _p(rOlh)_ - 0.0625 _p(fOrOlh) + 0.750 p(fOfOrOlh)._ This example illustrates that the projection vectors need not contain only positive entries. 

### 3 Conclusion 

We have introduced a predictive state representation for dynamical systems that is grounded in actions and observations and shown that, even in its linear form, it is at least as general and compact as POMDPs. In essence, we have established PSRs as a non-inferior alternative to POMDPs, and suggested that they might have important advantages, while leaving demonstration of those advantages to future work. We conclude by summarizing the potential advantages (to be explored in future work): 

Learnability. The k-order Markov model is similar to PSRs in that it is entirely based on actions and observations. Such models can be learned trivially from data by counting-it is an open question whether something similar can be done with a PSR. Jaeger (2000) showed how to learn such a model in the uncontrolled setting, but the situation is more complex in the multiple action case since outcomes are conditioned on behavior, violating some required independence assumptions. 

Compactness. We have shown that there exist linear PSRs no more complex that the minimal POMDP for an environment, but in some cases the minimal linear PSR seems to be much smaller. For example, a POMDP extension of _factored MDPs_ explored by Singh and Cohn (1998) would be cross-products of separate POMDPs and have linear PSRs that increase linearly with the number and size of the component POMDPs, whereas their minimal POMDP representation would grow as the size 

2If _U_ = A~BT is the singular value decomposition of _U,_ then B:E+ _AT_ is the pseudoinverse. The pseudoinverse of the diagonal matrix }J replaces each non-zero element with its reciprocal. 

of the state space, Le., exponential in the number of component POMDPs. This (apparent) advantage stems from the PSR's combinatorial or factored structure. As a vector of state _variables,_ capable of taking on diverse values, a PSR may be inherently more powerful than the distribution over discrete states (the belief state) of a POMDP. We have already seen that general PSRs can be more compact than POMDPs; they are also capable of efficiently capturing environments in the diversity representation used by Rivest and Schapire (1994), which is known to provide an extremely compact representation for some environments. 

Generalization. There are reasons to think that state variables that are themselves predictions may be particularly useful in learning to make other predictions. With so many things to predict, we have in effect a set or sequence of learning problems, all due to the same environment. In many such cases the solutions to earlier problems have been shown to provide features that generalize particularly well to subsequent problems (e.g., Baxter, 2000; Thrun & Pratt, 1998). 

Powerful, extensible representations. PSRs that predict tests could be generalized to predict the outcomes of multi-step options (e.g., Sutton et al., 1999). In this case, particularly, they would constitute a powerful language for representing the state of complex environments. 

AcknowledgIllents: We thank Peter Dayan, Lawrence Saul, Fernando Pereira and Rob Schapire for many helpful discussions of these and related ideas. 

### References 

Baum, L. E., Petrie, T., Soules, G., & Weiss, N. (1970). A maximization technique occurring in the statistical analysis of probabilistic functions of Markov chains. _Annals of Mathematical Statistics,_ 41, 164-171. 

- Baxter, J. (2000). A model of inductive bias learning. _Journal of Artificial Intelligence Research,_ 12, 149-198. 

- Chrisman, L. (1992). Reinforcement learning with perceptual aliasing: The perceptual distinctions approach. _Proceedings of the Tenth National Conference on Artificial Intelligence_ (pp. 183-188). San Jose, California: AAAI Press. 

- Jaeger, H. (2000). Observable operator models for discrete stochastic time series. _Neural Computation,_ 12, 1371-1398. 

- Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). Planning and acting in ' partially observable stochastic domains. _Artificial Intelligence, 101, 99-134._ 

- Lovejoy, W. S. (1991). A survey of algorithmic methods for partially observable Markov decision processes. _Annals of Operations Research,_ 28, 47-65. 

- McCallum, A. K. (1995). _Reinforcement learning with selective perception and hidden state._ Doctoral diss.ertation, Department of Computer Science, University of Rochester. 

- Rivest, R. L., & Schapire, R. E. (1994). Diversity-based inference of finite automata. _Journal of the ACM,_ 41, 555-589. 

- Shatkay, H., & Kaelbling, L. P. (1997). Learning topological maps with weak local odometric information~ _Proceedings of Fifteenth International Joint Conference on Artificial Intelligence (IJCAI-91)_ (pp. 920-929). 

- Singh, S., & Cohn, D. (1998). How to dynamically merge Markov decision processes. _Advances in Neural and Information Processing Systems 10_ (pp. 1057-1063). 

- Sutton, R. S., Precup, D., & Singh, S. (1999). Between MDPs and semi-MDPs: A frame- 

- work for temporal abstraction in reinforcement learning. _Artificial Intelligence, 181-211._ 

- Thrun, S., & Pratt, L. (Eds.). (1998). _Learning to learn._ Kluwer Academic Publishers. 

