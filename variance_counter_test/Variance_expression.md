The $X=\sum_{i=1}^N g_i$ is the accumulation of gradients $g_i$ for $i=1,2,3, \ldots, N$, where $N$ is our total count of observations.
Each $g_i$ is distributed according to same random variable G.
\
Given $$\operatorname{grad}=\sum_{i=1}^N g_i$$
$$
\operatorname{grad2} =\sum_{i=1}^N g_i^2
$$
We can have $$\operatorname{Var}(X)=\operatorname{Var}\left(\sum_{i=1}^N G\right)=\sum_{i=1}^N \operatorname{Var}(G)$$
Here we assume
each realization is independent.
\
Intuitively, whatever variance we measure at our samples/realizations  $g_i$ should be our variance at $X$ (within scaling factor). Hence $$\operatorname{Var}(X)=N \cdot \operatorname{Var}(G)$$
$$
\Rightarrow \operatorname{Var}(G)=E\left[\left(G-\mu_G\right)^2\right]=\sum_{i=1}^N\left(g_i-\mu_G\right)^2 p\left(g_i\right)
$$
We can assume that samples  $g_i$  are uniformly sampled.
\
Also, note that $\mu_G=\frac{1}{N}\operatorname{grad}$. Then we can have

$$\operatorname{Var}(G)=\frac{1}{N} \sum_{i=1}^N\left(g_i^2+\mu_G{ }^2-2 \mu_G g_i\right)$$
$$=\frac{1}{\dot{N}}\left(\sum_{i=1}^N g_i^2+\sum_{i=1}^N \frac{\mathrm{grad}^2}{N^2}-\frac{2}{N} \operatorname{grad} \sum_{i=1}^N g_i\right)$$
$$=\frac{1}{N}\left(\operatorname{grad2} +\frac{\operatorname{grad}^2}{N}-\frac{2}{N} \operatorname{grad}^2\right) $$
$$=\frac{1}{N}\left(\operatorname{grad2} -\frac{\operatorname{grad}^2}{N}\right)$$

\
Finally we can have
$$\operatorname{Var}(X)=\operatorname{Var}\left(\sum_{i=1}^N G\right)=\sum_{i=1}^N \operatorname{Var}(G)$$
$$=\operatorname{grad} 2-\frac{\operatorname{grad}^2}{N}$$
