### Latent variables

The normal mixture 
$$
f(x)=p_1Norm(\theta_1)+\cdots+p_kNorm(\theta_k)
$$
can be modelled using hierarchical model
$$
\begin{align}
& Z \sim multinomial(p_1,\ldots,p_k) \\
& (X | Z=j)\sim Normal(\theta_j)
\end{align}
$$
$P(Z=j)=p_j$, for $j=1, \ldots, k$.

We have a data set $\mathbf{X}$ with sample size $n$. 



Let's assume that $Z$ is also observed here.



The augmented likelihood will be 
$$
L(\theta | X, Z)=\prod^{k}_{j=1}\{\prod^{n}_{i=1,z_i=j}f_j(X_i|\mu_j)\}
$$
where $\prod^{n}_{i=1,z_i=j}f_j(X_i|\mu_j)$ means all $X$ that are from $j$th component.



The joint posterior pdf 
$$
P(\theta | X, Z) \propto \prod^{k}_{j=1}\{\prod^{n}_{i=1,z_i=j}f_j(X_i|\mu_j)\}\times priors
$$

What we need is an estimate of $\theta$ ,which is sampled from joint pdf $P(\theta | X, Z)$.

 We sample $\theta$ using Gibbs sampler from conditional distributions or the proportional ones, which can be simply found by setting other variables as constant in the joint pdf above.

- $\theta_1 \sim P(\theta_1 | \theta_{2tok}, X, Z)$

- $\cdots$
- $\theta_k \sim P(\theta_k | \theta_{1to (k-1)}, X, Z)$

### What if $Z$ is unknown

Our goal is to get $\theta$, but $Z$ is also unobserved.

This is actually easy, because $\theta$  can be also got by sampling from posterior joint distribution $P(\theta, Z |X)$ or any posterior distribution that contains $\theta$ and conditioned on $X$, but it does not have to be. Whether or not the joint posterior contains other variables besides of $\theta$ does not even matter here.

After dropping $Z$, this sample is equivalent with the one drawn from $P(\theta|X, Z)$.



**The conditional pdf of $Z$ has nothing to do with likelihood $L(\theta | X, Z)$, which is only used to get the conditional distribution of $\theta$.**

The conditional pdf of $Z$ has to be computed separately by using Bayes rule with extra conditions
$$
P(Z|X, \theta)= \frac{P(X|Z,\theta) \times P(Z | \theta)}{P(X|\theta)}
$$



Since $Z$ is discrete variable, its distribution has to be computed discretely for $j=1, \cdots, k$

-  $P(Z=j|\theta)=p_j$
- $P(X|Z=j,\theta)=Norm(\theta_j)$

- $P(X|\theta)=p_1Norm(\theta_1)+\cdots+p_kNorm(\theta_k)$

  

Now finally, we have all conditional distributions, $P(\theta |X, Z)$ as well as $P(Z|\theta, X)$.

The reason that we introduce latent variable is 

- make the conditional distribution $P(\theta |X, Z)$ simpler. By using latent variable, $P(\theta |X, Z)$ can be turned into natural distributions sometimes, making sampling much easier and more efficient.

- make the whole modelling more natural.

  

