
## IPIRec(Rev4) -- Estimator

### [IPIRecEstimatorSeries3](./ipirec_estimator_series3.py)
$$\begin{gather}
\hat{r}(u,i)=~ \frac{1}{|T(u)|}\sum_{x \in T(u)}\frac{\Pi_{y \in T(i)} \omega(u,x,y)s(x,y)}{\Pi_{y \in T(i)}|\omega(u,x,y)|} \nonumber \\
\epsilon(u,i) =~ r(u,i) - \hat{r}(u,i) \nonumber \\
s(x,y)=~ s(x,y)+\lambda^{S}
\eta \left(\epsilon(u,i)
\Big(\frac{s(x,y)}{\left\lVert S \right\rVert_{2}}
+\frac{\bar{s}(x)+\bar{s}(y)}{2}
\Big) \right) \nonumber \\
\eta(x)=~ \begin{cases}
x \geq 0,  2^{-1}x \\
\text{otherwise},  2x
\end{cases} \nonumber \\
\omega(u,x,y)=~ \omega(u,x,y)
\frac{r(u,i)+\lambda^{\Omega} s(x,y)}
{\hat{r}(u,i)+\gamma^{\Omega}\mathbb{W}(u,i)} \nonumber \\
\mathbb{W}(u,i)=~ \sum_{x\in NT(u)}\sum_{y \in T(i)} |\omega(u,x,y)| \nonumber \\
s(x,y)=~ s(x,y)\frac
{r(u,i)+\lambda^{\Omega}\omega(u,x,y)}
{\hat{r}(u,i) +\gamma^{\Omega}\mathbb{S}(u,i)} \nonumber \\
\mathbb{S}(u,i)=~ \sum_{NT(u)}\sum_{y \in T(i)} |s(x,y)| \nonumber
\end{gather}$$

- objective function

$$\begin{gather}
\mathbb{L}(\mathbb{X}, S, \Omega)=\text{RMSE}(\mathbb{X}, S, \Omega)=
\sqrt{\frac{1}{|\mathbb{X}|}\sum_{(u,i) \in \mathbb{X}} \left(r(u,i) - \hat{r}(u,i) \right)^{2}} \nonumber \\
\mathbb{L}^{S}(\mathbb{X}(a), S, \Omega) =~ \min \text{RMSE}^{S}(\mathbb{X}(a), S, \Omega) \nonumber \\
\mathbb{L}^{\Omega}(\mathbb{X}(a), S, \Omega) =~ \min \text{RMSE}^{\Omega}(\mathbb{X}(a), S, \Omega) \nonumber \\
\mathcal{L}(\mathbb{X}(a), S, \Omega) 
    =~ \frac{2 \mathbb{L}^{S}(\mathbb{X}(a), S, \Omega) \mathbb{L}^{\Omega}(\mathbb{X}(a), S, \Omega)}
    {\mathbb{L}^{S}(\mathbb{X}(a), S, \Omega) + \mathbb{L}^{\Omega}(\mathbb{X}(a), S, \Omega)} \nonumber \\
\min \mathcal{L}(\mathbb{X}, S, \Omega) \nonumber
\end{gather}$$

### [IPIRecEstimatorSeries4](./ipirec_estimator_series4.py)
- IPIRecEstimatorSeries3 목적함수에 일반화 항 $\mathcal{F}(\mathbb{X}(a), S, \Omega)$만 추가됨

$$\begin{gather}
\mathcal{F}(\mathbb{X}(a), S, \Omega)= \frac{|\mathbb{X}(a)|}{|U|^{2} |I|} \sum_{u \in U}\big( \frac{1}{|T(u)||T|} \sum_{x \in T(u)} \sum_{y \in T} \left( w(u,x,y)s(x,y) \right)^{2} \big)^{0.5} \nonumber \\
\mathbb{L}(\mathbb{X}, S, \Omega)=\text{RMSE}(\mathbb{X}, S, \Omega)+ \mathcal{F}(\mathbb{X}(a), S, \Omega) \nonumber
\end{gather}$$