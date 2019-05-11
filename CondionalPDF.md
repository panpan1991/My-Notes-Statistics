### The essence of conditional pdfs

Surprisingly, we have
$$
P(A|B)\propto P(B|A)\propto P(A,B)
$$


since
$$
P(A|B)=\frac{P(A,B)}{P(B)} \\
P(B|A)=\frac{P(A,B)}{P(A)} \\
$$
The only difference is the normalizing constant. In other words, the kernel of the conditional pdfs are completely same.

### How to use this property

For example, we have the joint pdf $f(x,y)$ 

- If we see $y$ as fixed constant, then $f(x, y_0)$ is the kernel of conditional pdf $f(x|y_0)$
- If we see $x$ as fixed constant, then then $f(x_0, y)$ is the kernel of conditional pdf $f(y|x_0)$

If we already have conditional pdf $f(x|y_0)$, and we want the kernel for condtional pdf $f(y|x_0)$.

We do not need to compute anything, instead, just need to treat the $x$ in $f(x|y_0)$ as fixed constant and treat $y_0$ as an random variable, then  $f(x|y_0)$ is indeed the kernel for $f(y|x_0)$.

In many situations, we only need the kernel of a conditional pdf.