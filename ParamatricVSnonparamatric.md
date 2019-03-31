## paramatric models are more specific than nonparamatric ones. They also require more information.

For example, Gaussian pdf with unknown mean and variance is a paramatric model. It can perfectly fit Gaussian data but cannot be used on data from other distributions.

Gaussian mixture model with infinite number of components is a nonparamatric model. It can be used to fit any data on real number set and without skewness.

But Gaussian mixture model with finite number of components is still a paramatric model since it cannot fit data from a mixture with more components than this modell. Though, we can still consider it as nonparamatric when the number of components is large enough, like 30. Because most data in real life cannot be from mixture with more than 30 components.

Nonparamatric model is much more universal and reuseful, since we do not know the distribution family in many cases.