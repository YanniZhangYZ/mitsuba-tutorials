$$X=\sum_{i=1}^N g_i$$ is the accumulation of gradients $g_i$ for $i=1,2,3, \ldots, N$, where $N$ is our total count of observations.
Each $g_i$ is distributed according to same random variable G.
\
Given $$grad=\sum_{i=1}^N g_i$$
$$grad2 =\sum_{i=1}^N g_i^2$$
We can have $$Var(X)=Var\left(\sum_{i=1}^N G\right)=\sum_{i=1}^N Var(G)$$
Here we assume
each realization is independent.
\
Intuitively, whatever variance we measure at our samples/realizations  $g_i$ should be our variance at $X$ (within scaling factor). Hence $$Var(X)=N \cdot Var(G)$$
$$\Rightarrow Var(G)=E\left[\left(G-\mu_G\right)^2\right]=\sum_{i=1}^N\left(g_i-\mu_G\right)^2 p\left(g_i\right)$$
We can assume that samples  $g_i$  are uniformly sampled.
\
Also, note that $\mu_G=\frac{1}{N}grad$. Then we can have

$$Var(G)=\frac{1}{N} \sum_{i=1}^N\left(g_i^2+\mu_G{ }^2-2 \mu_G g_i\right)$$
$$=\frac{1}{N}\left(\sum_{i=1}^N g_i^2+\sum_{i=1}^N \frac{\mathrm{grad}^2}{N^2}-\frac{2}{N} grad \sum_{i=1}^N g_i\right)$$
$$=\frac{1}{N}\left(grad2 +\frac{grad^2}{N}-\frac{2}{N} grad^2\right) $$
$$=\frac{1}{N}\left(grad2 -\frac{grad^2}{N}\right)$$

\
Finally we can have
$$Var(X)=Var\left(\sum_{i=1}^N G\right)=\sum_{i=1}^N Var(G)$$
$$=grad2-\frac{grad^2}{N}$$
