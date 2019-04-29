# Independence $\neq$ Uncorrelated

Since correlation only implies **Linear** dependence.

X and Y can be nonlinearly dependent but still with zero correlation.

An good example is $X\sim Normal(0,1)$ and $Y=X^2$. Apparently, they are very dependent, but their correlation will be zero.
$$
cov(X, Y)= E(XY)-E(X)E(Y) =E(X^3)-E(X)E(Y)=0-0=0
$$


#### In multivariate normal distribution, uncorrelated among components implies their independence.

$(X,Y)\sim N(\mu, \Sigma)$

In multivariate normal distribution, components do not have nonlinear dependence, since in the definition of multivariate normal, every component can be expressed with a linear combination of other components.



**Kendall's tau**



