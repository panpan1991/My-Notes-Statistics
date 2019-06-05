# Gibbs Sampler

Assume we have parameter $\theta_1, \theta_2, \theta_3$, and we use gibbs sampling to get MC chains. In the blue book, I saw the algorithm as

- $\theta_1^{(n+1)} \sim P(\theta_1|\theta_2^{(n)}, \theta_3^{(n)})$
- $\theta_2^{(n+1)} \sim P(\theta_2|\theta_1^{(n+1)}, \theta_3^{(n)})$
- $\theta_3^{(n+1)} \sim P(\theta_3|\theta_1^{(n+1)}, \theta_2^{(n+1)})$

The send of third steps of above used updated $\theta_1$ and $\theta_2$ as condition in the sampling of the $\theta_2$ and $\theta_3$.

But what if we use estimates from last loop only in the conditions, shown as below


- $\theta_1^{(n+1)} \sim P(\theta_1|\theta_2^{(n)}, \theta_3^{(n)})$
- $\theta_2^{(n+1)} \sim P(\theta_2|\theta_1^{(n)}, \theta_3^{(n)})$
- $\theta_3^{(n+1)} \sim P(\theta_3|\theta_1^{(n)}, \theta_2^{(n)})$

What will be the difference?

In my code for elliptical distribution, I see the sampling scheme of $\alpha$'s is using the second case instead of the first and it seems also working very well. 

