# What is MCMC(Monte Carlo Markov Chain)



In Bayesian Statistics, we write down the likelihood and priors. Their product is the joint posterior distribution of parameters.

In stead of point estimates or confidence intervals of parameters, Bayesian statistics gives us a joint distribution of parameters we are interested in.

By this distribution, we can draw a sample to get the point estimates.

However, the posterior can be in a dirty form or not even in a closed form. MCMC is a method to draw a sample from dirty form of pdf by drawing samples from conditional pdfs iteratively.



### Example of MCMC: Gibbs sampler

Two draw sample from $ f(x,y)$, we can do it in two steps:

- draw $x$ from $f(x|y)$ 
- draw $y$ from $f(y|x)$

since in many cases, $f(x|y)$ and $f(y|x)$ can be significantly simpler than $f(x, y)$. Of course, to launch the process, we have to set up an initial value for $y$.



