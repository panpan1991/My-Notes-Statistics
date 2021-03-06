### Elliptical distribution

$$
f(\mathbf{x};\Omega,g)=c_p|\Omega|^{-1/2}g((\mathbf{x})'\Omega^{-1}(\mathbf{x}))
$$

Given data $\mathbf{X}$, the joint likelihood function is 
$$
L(\Omega,g;\mathbf{X})=|\Omega|^{-1/2}g((\mathbf{x})'\Omega^{-1}(\mathbf{x}))
$$
The generator $g$ is not so easy to model, since it is not a pdf.

And the data is three dimensional.

### Conditional likelihood of $\alpha$ $\beta$ and $w$
If we assume $\Omega$ is known, we can get **latent variable **

$$
\mathbf{R}=\sqrt{(\mathbf{X})'{\Omega}^{-1}(\mathbf{X})}
$$
Why we can assume $\Omega$ is known?

That's because we only need **conditional likelihood** in MCMC method!
$$
L(g;\Omega,\mathbf{X}, R)=h(r)
$$
where $g$ is the generator function and it can be contained in $h(r)$.

$h(r)$ can be modeled as a gamma mixture, since it is actually a density function of $r$.

### Conditional likelihood of $\Omega$

$h(r)$ contain correlation matrix, so we still need to use the original pdf of $X$
$$
\begin{equation}
L(\Omega; \mathbf{\mu},\mathbf{x},g)=|{A}|^{-1}((\mathbf{X})'{\Omega}^{-1}(\mathbf{X}))^{(1-p)/2}h(\sqrt{(\mathbf{X})'{\Omega}^{-1}(\mathbf{X})}).
\end{equation}
$$




### Parameters need to be estimated

If a gamma mixture is used to model the function $h$, then it must contain parameters

- $\alpha$s
- $\beta$s
- $w$s, which are weights of the mixture

Besides, the correlation matrix $\Omega$ also needs to estimated.



### Posterior distributions needed

- $P(\alpha|\beta, w, \Omega,\mathbf{X}, Z)$ for each $\alpha$
- $P(\beta|\alpha, w,  \Omega,\mathbf{X}, Z)$ for each $\beta$
- $P(w|\alpha, \beta,  \Omega,\mathbf{X}, Z)$ for each $w$
- $P(\Omega|\alpha,\beta, w,\mathbf{X}, Z)$
- $P( Z|\alpha, \beta, w, \Omega,\mathbf{X})$ 

With those conditional posterior distributions, joint samples of $(\alpha, \beta, w, \Omega, Z)$ can be drawn by using Monte Carlo Method.

The first three posterior distributions become much simpler after introducing the latent variable, avoiding dirty form like multiplication of summation.

$R$ is not latent variable, but simply another form of data set with known $\Omega$. Every time we update $R$ inside the loop is just making posterior distribution of $\alpha$, $\beta$ and $w$ conditioned on the current update of $\Omega$.

$\alpha$'s are mutually independent, so are $\beta$'s. This means $\alpha_j$ 's posterior distribution does not have to be conditioned on other $\alpha$'s. 

