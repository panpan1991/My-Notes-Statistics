### Introduction of copula

- In multivariate Statisitcs, Copula is a linking function between marginal CDFs and the joint CDF.
- Copula itself means **linking** originally.
- The simplest copula is multiplication function, when multivariate components are independent.
- Copula models the dependence structure (**correlations**) of among variables without determining marginal distributions.
- Copula can therefore be used as a tool for devising new multivariate distributions.

### How to model a copula

Many copulas are modeled with CDF and Quantile functions.

Copula is a linking function taking marginal CDF values as input and returning a joint CDF value. This can be done by function composition of CDF and Quantile functions.

$C(u_1, u_2)=F(Q(u_1), Q(u_2))$

