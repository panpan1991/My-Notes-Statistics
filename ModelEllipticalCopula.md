# Model Elliptical copula

### Data structure

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
&=\prod_{i=1}^{n}|{A}|^{-1}({r_i^2})^{(1-p)/2}h(({r_i^2})^{1/2})\\
&=\prod_{i=1}^{n}|{A}|^{-1}({r_i})^{(1-p)}h({r_i}) \\
&=\prod_{i=1}^{n}|{A}|^{-1}({r_i})^{(1-p)}\prod_{j=1}^{k}(w_jf(r_i|\alpha_j,\beta_j)^{z_{ij}}) 
\end{align}
$$
where 
$$
\begin{align}r_i^2 &=(\mathbf{x})'{\Omega}^{-1}(\mathbf{x}) \\&=(Q_{EC}(u_{i1}), Q_{EC}(u_{i2})){\Omega}^{-1}(Q_{EC}(u_{i1}), Q_{EC}(u_{i2}))'\\
\\
X&=(Q_{EC}(u_{i1}), Q_{EC}(u_{i2}))
\end{align}
$$

In this likelihood, **both $h$ function and quantile function $Q_{EC}$ contain parameters $\alpha$, $\beta$ as well as weights $w$**. 

### Conditional likelihood

- $L(\alpha, \beta, w| \Omega, U, X,  R)$
- $L(\Omega|\alpha, \beta, w,U, X, R)$

These two likelihood above need the whole pdf involved. 

**$\alpha$'s are not mutually independent. Every time we update $\alpha_j$, $R$ has to be updated to make sure next operations are conditioned on the current estimate of $\alpha_j$. This requirement also applies on the estimation of $\beta$ as well as weights**.



### Log likelihood

$$
\begin{align}
&L(\mathbf{ \alpha }, \mathbf{\beta}, \mathbf{w}, A; \mathbf{U}) \\
&=\prod_{i=1}^{n}c(u_{i1}, u_{i2}) \\
&=\prod_{i=1}^{n}|{A}|^{-1}({r_i^2})^{(1-p)/2}h(({r_i^2})^{1/2})\\
&=\prod_{i=1}^{n}|{A}|^{-1}({r_i})^{(1-p)}h({r_i}) \\
&=\prod_{i=1}^{n}\{|{A}|^{-1}({r_i})^{(1-p)}\prod_{j=1}^{k}(w_jf(r_i|\alpha_j,\beta_j))^{z_{ij}}\} \\
&=-nlog|A|+(1-p)\sum_{i=1}^{n} log(r_i)+\sum_{i=1}^{n}log\Big(\prod_{j=1}^{k}(w_jf(r_i|\alpha_j,\beta_j))^{z_{ij}}\Big)\\
&=-nlog|A|+(1-p)\sum_{i=1}^{n} log(r_i)+\sum_{i=1}^{n}\sum_{j=1}^{k}log\Big(w_jf(r_i|\alpha_j,\beta_j)\Big)^{z_{ij}}\\
&=-nlog|A|+(1-p)\sum_{i=1}^{n} log(r_i)+\sum_{i=1}^{n}\sum_{j=1}^{k}{z_{ij}}\Big(log(w_j)+log\big(f(r_i|\alpha_j,\beta_j)\big)\Big)\\
&=-nlog|A|+(1-p)\sum_{i=1}^{n} log(r_i)+\sum_{i=1}^{n}\sum_{j=1}^{k}\Big({z_{ij}}log(w_j)+{z_{ij}}log\big(f(r_i|\alpha_j,\beta_j)\big)\Big)\\
&=-nlog|A|+(1-p)\sum_{i=1}^{n} log(r_i)+\sum_{i=1}^{n}\sum_{j=1}^{k}\Big({z_{ij}}log(w_j)\Big)+\\
&\sum_{i=1}^{n}\sum_{j=1}^{k}\Big({z_{ij}}log\big(f(r_i|\alpha_j,\beta_j)\big)\Big)\\
\end{align}
$$



### Priors

For $\alpha$'s
$$
\frac{c}{\alpha_j^{c+1}}
$$
For $\beta$'s
$$
\frac{b^a}{\Gamma(a)}\beta_j^{a-1}e^{-b\beta_j}
$$
For $v$'s
$$
\gamma(1-v_j)^{\gamma-1}
$$
For $\gamma$
$$
\gamma^{\eta_1-1}e^{-\eta_2\gamma}
$$
For $\rho$'s
$$
\mathbf{1}_{(-1 \leq \rho \leq 1)}
$$


### Log priors
$$
\begin{align}
&\sum_{j=1}^{k}\big(log(c)-(c+1)log(\alpha_j)\big) \\
& + \sum_{j=1}^{k}\big(alog(b)+(a-1)log(\beta_j)-b\beta_j-logamma(a)\big)\\
& + \sum_{j=1}^{k-1}\big(log(\gamma)+(\gamma-1)log(1-v_j)\big)\\
& + (\eta_1-1)log(\gamma)-\eta_2\gamma
\end{align}
$$



### Conditional log posteriors for MCMC process

**$\alpha$ from different components are not mutually independent any more. and $\beta$'s and weights are also like this. This cause sampling much more difficult and time consuming.** 

- for $\alpha_j$
$$
\begin{align}
  &(1-p)\sum_{i=1}^{n} log(r_i)+
  \sum_{i=1}^{n}\sum_{j=1}^{k}\Big({z_{ij}}log\big(f(r_i|\alpha_j,\beta_j)\big)\Big) +\sum_{j=1}^{k}(c+1)log(\alpha_j)\\
  \end{align}
$$

- for $\beta_j$
$$
\begin{align}
  &(1-p)\sum_{i=1}^{n} log(r_i)+\sum_{i=1}^{n}\sum_{j=1}^{k}\Big({z_{ij}}log\big(f(r_i|\alpha_j,\beta_j)\big)\Big) + \\
  &\sum_{j=1}^{k}\big((a-1)log(\beta_j)-b\beta_j\big)\\
  \end{align}
$$

- for $v_j$
$$
\begin{align}
  &(1-p)\sum_{i=1}^{n} log(r_i)+\sum_{i=1}^{n}\sum_{j=1}^{k}\Big({z_{ij}}log(w_j)\Big)+\\
  &\sum_{i=1}^{n}\sum_{j=1}^{k}\Big({z_{ij}}log\big(f(r_i|\alpha_j,\beta_j)\big)\Big) + \sum_{j=1}^{k-1}\big((\gamma-1)log(1-v_j)\big)\\
  \end{align}
$$

- for $\theta$
$$
\begin{align}
  &-nlog|A|+(1-p)\sum_{i=1}^{n} log(r_i)+
  \sum_{i=1}^{n}\sum_{j=1}^{k}\Big({z_{ij}}log\big(f(r_i|\alpha_j,\beta_j)\big)\Big)
  \end{align}
$$


Using Metropolis or HMC to draw samples from these log posterior pdfs.

### Conditional posterior distributions for latent variable

- for latent variables $Z$
$$
  \begin{align}
  P(\mathbf{Z}|\alpha,\beta, \mathbf{V},\gamma,\rho,\mathbf{R}) 
  & = \frac{P(\mathbf{R}|\alpha,\beta, \mathbf{V},\gamma,\rho,\mathbf{Z}) \times P(\mathbf{Z}) }{P(\mathbf{R}|\alpha,\beta, \mathbf{V},\gamma,\rho) }

  \end{align}
$$

$$
\begin{align}
  r_i^2 &=(\mathbf{x})'{\Omega}^{-1}(\mathbf{x}) 
  \\&=(Q_{EC}(u_{i1}), Q_{EC}(u_{i2})){\Omega}^{-1}(Q_{EC}(u_{i1}), Q_{EC}(u_{i2}))'\\
  &=Q_{EC}(\mathbf{U})
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