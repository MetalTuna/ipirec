
## IPIRec(Rev1) -- Estimator
$$\begin{gather}
\hat{r}(u,i) = \frac{\sum_{x}\sum_{y} \omega(u,x,y) s(x,y) \nu(u,y)}{\sum_{x \in NT(u)}\sum_{y \in T(i)} |\omega(u,x,y)|} \nonumber \\
\nu(u,y) = \begin{cases}
NT(u) \cap y = \phi & DV \\
\text{otherwise} & 1 
\end{cases} \nonumber \\
\epsilon(u,i) = r(u,i) - \hat{r}(u,i) \nonumber \\
w(x,y) = \frac{w + \eta \left(w(x,y) \epsilon(u,i) \lambda^{S} \gamma^{S} s(x,y)\right)}{\sum_{z \in T(i)} |w(x,z) s(x,z)|} \nonumber \\
\eta (x) = \begin{cases}
x \geq 0 & 2^{-1}x \\
\text{otherwise} & 2x
\end{cases} \nonumber \\
S = S \cdot W  \nonumber\\
\mathbb{W}(u) = \sum_{x \in NT(u)}\sum_{y \in T(i)} |\omega(u,x,y)| \nonumber \\
\omega(u,x,y) = \omega(u,x,y) \frac{r(u,i) + \lambda^{\Omega}s(x,y)}{\hat{r}(u,i) + \gamma^{\Omega} \mathbb{W}(u)} \nonumber
\end{gather}$$
