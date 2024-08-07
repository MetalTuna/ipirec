
## IPIRec(Rev2) -- Estimator

$$\begin{gather}
\hat{r}(u,i) = \frac{\sum_{x}\sum_{y} \omega(u,x,y) s(x,y) \nu(u,y)}{\sum_{x \in NT(u)}\sum_{y \in T(i)} |\omega(u,x,y)|} \nonumber \\
\nu(u,y) = \begin{cases}
NT(u) \cap y = \phi & DV \\
\text{otherwise} & 1 
\end{cases} \nonumber \\
\epsilon(u,i) =~ r(u,i) - \hat{r}(u,i) \nonumber \\
s(x,y)=~ s(x,y)+\lambda^{S}
\eta \left(\epsilon(u,i) \Big(\frac{s(x,y)}{\left\lVert S \right\rVert _{2}} \Big) \right) + \gamma^{S} \frac{\bar{s}(x)+\bar{s}(y)}{2} \nonumber \\
\eta(x)=~ 
\begin{cases}
x \geq 0,  2^{-1}x \\
\text{otherwise},  2x
\end{cases} \nonumber \\
\end{gather}$$

$$\begin{gather}
\alpha(x) = 
\begin{cases}
x = 0 & x+1 \\
\text{otherwise} & x 
\end{cases} \nonumber \\ 
\omega(u,x,y)=~ \omega(u,x,y)\frac{r(u,i)+\lambda^{\Omega} s(x,y)}{\hat{r}(u,i)+\gamma^{\Omega}\mathbb{W}(u,i)} \nonumber \\
\mathbb{W}(u,i)=~ \sum_{x \in NT(u)}\sum_{y \in T(i)} |\omega(u,x,y)| \nonumber \\
s(x,y)= s(x,y)\frac{r(u,i)+\alpha(\hat{r}(u,i))+\lambda^{\Omega}\omega(u,x,y)}{\alpha(\hat{r}(u,i)) +\gamma^{\Omega}\mathbb{S}(u,i)} \nonumber \\
\mathbb{S}(u,i)=~ \sum_{NT(u)}\sum_{y \in T(i)} |s(x,y)| \nonumber 
\end{gather}$$
