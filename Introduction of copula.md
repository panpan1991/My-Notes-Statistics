

### Introduction of copula

- In multivariate Statisitcs, Copula is a linking function between marginal CDFs and the joint CDF.
- Copula itself means **linking** originally.
- The simplest copula is multiplication function, when multivariate components are independent. It is called Farlie - Gumbel - Morgenstern (FGM) copula.
- Copula models the dependence structure (**correlations**) of among variables without determining marginal distributions.
- Copula can therefore be used as a tool for devising new multivariate distributions.



### The essence of copula

Copula is just a fancy name for a multivariate distribution with:

- standard **Uniform** margins 
- specific correlation capturing the dependence between marginal variables. 

It has to be mentioned that multivariate distribution with uniform margins **should not** be written as Multivariate Uniform.



In other words, so called Copula is just a distribution：

- for  random variable from **probability integral transformation** 
  $$
  u_i=F_i(x_i)
  $$

- return joint probability as output



### How to model a copula

Since copula is a distribution, we can model its CDF or PDF. The former is easier and more straightforward.

Many copulas' CDF are modeled with CDF and Quantile functions.

This can be done by function composition of **multivariate CDF** and **univariate Quantile functions**.

$$
C(u_1, u_2)=F(Q(u_1), Q(u_2))
$$



For example, Gaussian copula uses CDF of multivariate and quantile function for standard normal.



### How to use a copula

Since copula is multivariate distribution with Uniform margins, as long as we already get the estimated marginal CDFs of a multivariate distribution, we can get joint probability by doing

- $$
  u_i=F_i(x_i), i=1,...p
  $$

- $$
  F(x_1,...x_p)=C(u_1,...u_p)
  $$

Here we are using the CDF of copula



To compute the pdf of the multivariate distribution that we are interested in, we can compute the derivative of $F(x_1,...x_p)$ by using chain rule. It is obviously made from a copula pdf and a Jacobian.
$$
f(x_1,...x_p)=c(u_1,...u_p)\times \frac{dU}{dX}
$$


In elliptical copula, CDF and margins of elliptical distributions are used in copula construction
$$
C(u_1, u_2)=F_{EC}(Q_{EC}(u_1), Q_{EC}(u_2))
$$


Nowadays, elliptical copulas with a specific corresponding elliptical distribution have been fully developed. 

Any multivariate distributions that can be modelled by an elliptical copula is called Meta-elliptical distribution.
$$
\begin{align}
f(x_1, x_2)
&=f_{EC}(Q_{EC}(u_1), Q_{EC}(u_2))\times \frac{dU}{dX} \\
&=f_{EC}(z_1, z_2)\times \frac{dU}{dX} 
\end{align}
$$
We already have models for $f_{EC}$, which is an elliptical distribution's cdf. When we know what the specific elliptical distribution is, in other words, when we know what $Q_{EC}$ is, modelling the copula would be equivalent to modelling an elliptical pdf.



### The drawbacks of Gaussian copula

### Elliptical Copula

Elliptical copula is a copula without specifying what elliptical family is used in the copula.
$$
\begin{align}c(u_1, u_2)&=f_{EC}(Q_{EC}(u_1), Q_{EC}(u_2))\times \frac{dU}{dX} \\&=f_{EC}(z_1, z_2)\times \frac{dU}{dX} \end{align}
$$


$dU/dX$ can be treated as known, since they can be estimated separately. 

Now we start model the key part $f_{EC}(Q_{EC}(u_1), Q_{EC}(u_2))$.

#### Stochastic form
A $p$ variate vector $\mathbf{X}$ from $\epsilon_p(\mathbf{\mu}, {\Omega}, g)$ can also be expressed in the following stochastic representation
$$
\mathbf{X}=\mathbf{\mu}+RA\mathbf{U}, 
$$
Given that  $\mathbf{\mu}=\mathbf{0}$ ,
$$
\mathbf{X}=RA\mathbf{U} \\
R^2=(\mathbf{X})'{\Omega}^{-1}(\mathbf{X}) \\
R=\sqrt{(\mathbf{X})'{\Omega}^{-1}(\mathbf{X})}
$$


In other words, elliptical distribution's pdf can be written as
$$
f(\mathbf{X};A)=\frac{\Gamma(p/2)}{2\pi^{p/2}}|{A}|^{-1}r^{1-p}h(r).
$$


This is pdf of $\mathbf{X}$ expressed with $r$, but this is not a pdf of $\mathbf{R}$, which needs a Jacobian multiplier. 



#### Meta-elliptical data

Assume we get bivariate data
$$
\mathbf{x}^*=(x_1^*, x_2^*)
$$
If we already estimated their marginal distributions, we can do an Probability Integral transformation:
$$
\mathbf{u}=(u_1, u_2)=(F_1(x_1^*), F_2(x_2^*))
$$
If we keep transformation, we can get 
$$
\mathbf{x}=(x_1,x_2)=(Q(u_1),Q(u_2))=(Q(F_1(x_1^*)), Q(F_2(x_2^*))
$$
$Q$ is the marginal quantile function of an elliptical distribution.

However, this time we do not know what the $Q$ exactly is.


The above is a demo for bivariate case. For trivariate case, $Q$ 's inverse function, the marginal CDF, can be written as a close form
$$
\begin{align*}
P(z)
&=1/2+\\
&\pi^{-1/2}\Gamma(3/2) \times\sum \frac{w_j}{(\alpha_j-1)\Gamma(\alpha_j-1)} \\
&\bigg[{z/|z|*\Gamma(\alpha_j)+\beta_j z\Gamma(\alpha_j-1, \beta_j |z|)-z/|z|*\Gamma(\alpha_j, \beta_j |z|)}\bigg]
\end{align*}
$$
Then $Q$ becomes a numerical function with parameters $(\mathbf{ \alpha }, \mathbf{\beta}, \mathbf{w})$.



Let's go back to the elliptical copula, which is constructed based on the pdf of an elliptical distribution.
$$
\begin{align} 
c(u_1, u_2)&=f_{EC}(Q_{EC}(u_1), Q_{EC}(u_2))\\
\\
&=f_{EC}((Q_{EC}(u_1), Q_{EC}(u_2));A) \\
\\
&=\frac{\Gamma(p/2)}{2\pi^{p/2}}|{A}|^{-1}r^{1-p}h(r) \\
\\
&=\frac{\Gamma(p/2)}{2\pi^{p/2}}|{A}|^{-1}({r^2})^{(1-p)/2}h(({r^2})^{1/2}) \\
\\
&=\frac{\Gamma(p/2)}{2\pi^{p/2}}|{A}|^{-1}({r^2})^{(1-p)/2}h(({r^2})^{1/2})
\end{align}
$$
where
$$
\begin{align}
r^2 &=(\mathbf{x})'{\Omega}^{-1}(\mathbf{x}) \\
&=(Q_{EC}(u_1), Q_{EC}(u_2)){\Omega}^{-1}(Q_{EC}(u_1), Q_{EC}(u_2))'
\end{align}
$$
From above, we can see $R^2$ is related to $Q$, which has $(\mathbf{ \alpha }, \mathbf{\beta}, \mathbf{w})$ involved.

**This makes the estimation in elliptical copula very different with that on a pure elliptical distribution.** Because the conditional distribution of parameters are significant different.



