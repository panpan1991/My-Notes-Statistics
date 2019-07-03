# Complete Quadratic Form 

In deriving posterior pdf for Multivariate Normal distribution, sometimes we need to complete a quadratic form.

PDF of multivariate normal is 
$$
f(\mathbf{X})\propto e^{-\frac{1}{2}(\mathbf{X}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{X}-\boldsymbol{\mu})}
$$
By some algebraic operations, the quadratic form in exponent can be expanded as
$$
-\frac{1}{2}\mathbf{X}^T\Sigma^{-1}\mathbf{X}+\boldsymbol{\mu}^T\Sigma^{-1}\mathbf{X}-\frac{1}{2}\boldsymbol{\mu}^T\Sigma^{-1}\boldsymbol{\mu}
$$
The last term $\boldsymbol{\mu}^T\Sigma^{-1}\boldsymbol{\mu}$ is not very useful for completing quadratic form, we need only terms with $\mathbf{X}$ involved.

Now suppose we have expanded quadratic form as
$$
-\frac{1}{2}\mathbf{X}^TA\mathbf{X}+B\boldsymbol{X}
$$
It is obvious that 

- $\Sigma=A^{-1}$

- $\boldsymbol{\mu}^T\Sigma^{-1}=B$, Thus, $\boldsymbol{\mu}=\Sigma B^T$. Notice that $\Sigma^T=\Sigma$.



