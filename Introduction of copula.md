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



#### The likelihood

Assume we get data set with size $n$
$$
\begin{bmatrix}
x_{11}, x_{12}, x_{13} \\
x_{21}, x_{22}, x_{23} \\
......\\
x_{n1}, x_{n2}, x_{n3} \\
\end{bmatrix}
$$


After probability integral transformation
$$
\begin{bmatrix}
u_{11}, u_{12}, u_{13} \\
u_{21}, u_{22}, u_{23} \\
......\\
u_{n1}, u_{n2}, u_{n3} \\
\end{bmatrix}
$$

The likelihood is
$$
\begin{align}
L(\mathbf{ \alpha }, \mathbf{\beta}, \mathbf{w}, A; \mathbf{U}) 
&=\prod_{i=1}^{n}c(u_{i1}, u_{i2}) \\
&=\prod_{i=1}^{n}|{A}|^{-1}({r_i^2})^{(1-p)/2}h(({r_i^2})^{1/2})
\end{align}
$$
where 
$$
\begin{align}r_i^2 &=(\mathbf{x})'{\Omega}^{-1}(\mathbf{x}) \\&=(Q_{EC}(u_{i1}), Q_{EC}(u_{i2})){\Omega}^{-1}(Q_{EC}(u_{i1}), Q_{EC}(u_{i2}))'\end{align}
$$

In this likelihood, both $h$ function and quantile function $Q_{EC}$ contain parameters $\alpha$, $\beta$ as well as weights $w$. 

### Conditional probabilities I



In my master thesis, I used a gamma mixture to model function $h$, and I had
$$
P(\beta_j|\alpha_j, \rho, \mathbf{r}, \mathbf{z}_j) \propto  \beta_j^{\alpha_j \sum_{i=1}^{n}z_{ij}+a-1 }e^{-\beta_j(\sum_{i=1}^{n}r_iz_{ij}+b)}.
$$
where $\beta_j$ is a parameter for $j$th component in the gamma mixture.

However, this is assuming that $r_i$'s do not contain $\beta$ .



In the new case, we have 
$$
\begin{align}r_i^2 &=(\mathbf{x})'{\Omega}^{-1}(\mathbf{x}) \\&=(Q_{EC}(u_{i1}), Q_{EC}(u_{i2})){\Omega}^{-1}(Q_{EC}(u_{i1}), Q_{EC}(u_{i2}))'\end{align}
$$
where $Q_{EC}$, the marginal quantile function of the elliptical distribution, is also modelled with the same gamma mixture that we used in modelling function $h$.



In joint posterior distribution
$$
\begin{aligned}
P(\mathbf{\alpha}, \mathbf{\beta}, \mathbf{V}, \gamma, \rho|\mathbf{U}, Z)
	& \propto |A|^{-n}\prod^n_{i=1}\left[r_i^{1-p}\prod_{j=1}^{k}\bigg(w_j\frac{\beta_j^{\alpha_j}}{\Gamma(\alpha_j)}r_i^{\alpha_j-1}e^{-\beta_jr_i}\bigg)^{z_{ij}}\right] \\	
	& \times \prod_{j=1}^{k}\frac{c}{\alpha_j^{c+1}} \\
	& \times \prod_{j=1}^{k}\frac{b^a}{\Gamma(a)}\beta_j^{a-1}e^{-b\beta_j} \\
	& \times \prod_{j=1}^{k-1}\gamma(1-v_j)^{\gamma-1} \\
	& \times \gamma^{\eta_1-1}e^{-\eta_2\gamma}\\
	& \times \mathbf{1}_{(-1 \leq \rho \leq 1)},
\end{aligned}
$$
In the right side of pdf, $\mathbf{U}$ is written in a form of $\mathbf{R}$. s

We won't able to separate a nice closed form posterior distribution of $\beta$, since its prior is **no longer conjugate** for the new likelihood having $\beta$ involved in $r_i$.



 The estimate of $\beta$ has to be done in a less elegant way, taking the dirty form of likelihood and using Metropolis method.
$$
\begin{align}
P(\mathbf{\beta} |\mathbf{\alpha}, \mathbf{V}, \gamma, \rho,\mathbf{U}, Z)
	& \propto \prod^n_{i=1}\left[r_i^{1-p}\prod_{j=1}^{k}\bigg(\frac{\beta_j^{\alpha_j}}{\Gamma(\alpha_j)}r_i^{\alpha_j-1}e^{-\beta_jr_i}\bigg)^{z_{ij}}\right] \\	
	& \times \prod_{j=1}^{k}\frac{b^a}{\Gamma(a)}\beta_j^{a-1}e^{-b\beta_j} \\
\end{align}
$$
The old posterior distribution for $\alpha$ is also wrong, since it didnot contain $\alpha$ in $r_i$'s.



### Conditional distributions II

- for latent variables $Z$
  $$
  \begin{align}
  P(\mathbf{Z}|\alpha,\beta, \mathbf{V},\gamma,\rho,\mathbf{X}) 
  & = \frac{P(\mathbf{X}|\alpha,\beta, \mathbf{V},\gamma,\rho,\mathbf{Z}) \times P(\mathbf{Z}) }{P(\mathbf{X}|\alpha,\beta, \mathbf{V},\gamma,\rho) }
  
  \end{align}
  $$
  In this case, $Z_i$'s are independent, of course, since they only depend on the corresponding $X_i$, thus the joint posterior distribution of $\mathbf{Z}$ is actually a set of independent univariate multinomial distributions. 

  In our case, the latent variable is defined as 
  $$
  Z_{ji}=1, \mbox{if }r_i \mbox{is from component }j
  $$
  we will have $30 \times n$ matrix
  $$
  \begin{bmatrix}
  1 0 1 \cdots 0 \\
  0 1 0 \cdots 0 \\
  \vdots \ddots \vdots \\
  0 0 0 \cdots 1 \\
  \end{bmatrix}
  $$
  which only has a single $1$ in its each column. Each column is a multinomial distribution.
  $$
  \begin{align}
  P(Z_{ji}=1|\alpha,\beta, \mathbf{V},\gamma,\rho,\mathbf{X}) 
  & = \frac{P(\mathbf{X}|\alpha,\beta, \mathbf{V},\gamma,\rho,Z_{ji}=1) \times P(Z_{ji}=1) }{P(\mathbf{X}|\alpha,\beta, \mathbf{V},\gamma,\rho) }
  \end{align}
  $$
  
- for $\alpha$'s 
  $$
  \begin{align}P(\mathbf{\alpha}|\mathbf{\beta} , \mathbf{V}, \gamma, \rho,\mathbf{X}, Z)    & \propto \prod^n_{i=1}\left[r_i^{1-p}\prod_{j=1}^{k}\bigg(\frac{\beta_j^{\alpha_j}}{\Gamma(\alpha_j)}r_i^{\alpha_j-1}e^{-\beta_jr_i}\bigg)^{z_{ij}}\right] \\       & \times \prod_{j=1}^{k}\frac{c}{\alpha_j^{c+1}} 
  \end{align}
  $$

- for $\beta$'s
  $$
  \begin{align}P(\mathbf{\beta} |\mathbf{\alpha}, \mathbf{V}, \gamma, \rho,\mathbf{X}, Z)    & \propto \prod^n_{i=1}\left[r_i^{1-p}\prod_{j=1}^{k}\bigg(\frac{\beta_j^{\alpha_j}}{\Gamma(\alpha_j)}r_i^{\alpha_j-1}e^{-\beta_jr_i}\bigg)^{z_{ij}}\right] \\       & \times \prod_{j=1}^{k}\frac{b^a}{\Gamma(a)}\beta_j^{a-1}e^{-b\beta_j} \\\end{align}
  $$
  
- for weights $w$'s

  Since weights are not independent with each other and they apparently sum up to $1$, we do not do inference on $w$ directly. Weights $w$ will be transformed into $v$ by using stick-breaking process

- for $v$'s
  $$
  \begin{aligned}
  P(\mathbf{\alpha}, \mathbf{\beta}, \mathbf{V}, \gamma, \rho|\mathbf{X}, Z)
  	& \propto\prod^n_{i=1}\left[r_i^{1-p}\prod_{j=1}^{k}\bigg(w_jr_i^{\alpha_j-1}e^{-\beta_jr_i}\bigg)^{z_{ij}}\right] \\	
  	& \times \prod_{j=1}^{k-1}\gamma(1-v_j)^{\gamma-1} \\
  \end{aligned}
  $$
  