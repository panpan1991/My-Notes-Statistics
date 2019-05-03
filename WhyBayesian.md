### Why Bayesian

In statistical inference, we are trying to locate the parameters of a specific distribution. 

In classic statistics, frequentists believe that the "statistic" must follow a sampling distribution with the "parameter" as its mean.

They build **confidence intervals** of the parameter according to the sampling distribution of the statistic, Or using the center of the interval as an estimate of the parameter.

This works very well if we only have one parameter to infer. However, when we have more than two parameters, we would have to build joint confidence interval, such as Bounferroni intervals, and the confidence level can be unsure. 



Bayesian statistics is for inference of high dimensional parameter. In many nonparametric statistical estimation, it will have more hundreds of parameters to infer. Classic methods can hardly deal with such issue. 

Bayesian method is about 

1. finding the **joint** posterior distribution of parameters 
2. draw samples from the distribution above
3. take an average and use as the estimate

The first step is usually easy. The focus is usually the second step, since most posterior distributions are not natural distributions and some of them do not even have closed form. The Monte Carlo Marlkov Chain, the **MCMC**, is a popular method to draw samples from a **dirty joint pdf**.

These three steps of Bayesian method usually apply everywhere, no matter how many parameters we have in total, this is why Bayesian method is so popular.



### The essence of likelihood function

Why the first step of Bayesian method is usually easy? Because it is just the product of likelihood and those priors.



### The essence of conditional distribution

Assume $f(x, y)$ is the joint pdf, fix one random variable, such as $x=x_0$, the $f(x_0, y)$ is proportional to conditional distribution $f(y|x=x_0)$, since
$$
f(y|x=x_0)=\frac{f(x_0, y)}{f(x=x_0)} \propto f(x_0, y)
$$
The denominator above is just a normalizing constant, which can be ignored if we only use $f(y|x=x_0)$ in likelihood of Bayesian method.

Thus, given a joint pdf, the conditional pdf can be got by simply setting other random variables into a fixed value, the "conditions". 



Assume we have 
$$
f(\theta | X, Z)
$$
we want 
$$
f(\theta, Z|X)
$$
we can do 
$$
f(\theta, Z|X)=f(\theta|X,Z) \times f(Z)
$$


























### Find the conditional distribution with corresponding joint distribution



