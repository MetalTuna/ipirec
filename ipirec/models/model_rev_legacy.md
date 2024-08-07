
## IPIRec(Rev 1, 2) -- Model (Series 1, 2)

### [IPIRecModelSeries1](./ipirec_model_series1.py)
$$\begin{gather}
s(x,y)=\text{CoOccur}(x,y)\cdot \Pr(x,y)\cdot 2^{-1}\sum_{d \in [U, I]}sim_{d}(x,y) \nonumber \\
\text{CoOccur}(x,y)=\frac{\ln(\min (|I(x) \cap I(y)| + 1,~ \theta_{I}))}{\ln(\max_{x', y' \in T} |I(x') \cap I(y')|)};~\theta_{I}=20~(\text{optional}) \nonumber \\
\Pr(x,y)=|I(x)|^{-1}|I(x)\cap I(y)| \nonumber \\ 
sim_{U}(x,y)=\frac{\sum_{u \in U(x) \cap U(y)}(r(u,x)-\bar{r}(x))(r(u,y)-\bar{r}(y))}{\sqrt{\sum_{u}(r(u,x)-\bar{r}(x))^{2}\sum_{u}(r(u,y)-\bar{r}(y))^{2}}} \nonumber \\
sim_{I}(x,y)=\frac{\sum_{i \in I(x) \cap I(y)}(r(i,x)-\bar{r}(x))(r(i,y)-\bar{r}(y))}{\sqrt{\sum_{i}(r(i,x)-\bar{r}(x))^{2}\sum_{i}(r(i,y)-\bar{r}(y))^{2}}} \nonumber \\
\bar{r}(x)= \frac{\sum_{u}|I(u)\cap I(V)|}{|U(x)\cap U(V)|}+\frac{\sum_{i}|U(i)\cap U(V)|}{|I(x) \cap I(V)|} \nonumber
\end{gather}$$

### [IPIRecModelSeries2](./ipirec_model_series2.py)
$$\begin{gather}
S \leftarrow S + \mathbb{B}(\mathbb{X}(V),S), \nonumber \\ 
s(x,y) = s(x,y) + \bar{b}(\mathbb{X}(V),S,x,y). \nonumber \\
\bar{b}(\mathbb{X}(V),x,y) =~ \hat{b}^{U}(\mathbb{X}(V),x,y) + \hat{b}^{I}(\mathbb{X}(V),x,y) + \frac{1}{2}\left(b(\mathbb{X}(V),x) + b(\mathbb{X}(V),y)\right) \nonumber \\
b(\mathbb{X}(V),t) =~ \frac{|R(t)|}{|R|} - \mu(\mathbb{X}(V)) \nonumber \\
\hat{b}^{U}(\mathbb{X}(V),x,y) =~ \frac{\sum_{u} b(\mathbb{X}(V),u)}{|U(x) \cup U(y)|} \nonumber\\
b(\mathbb{X}(V),u) =~ \frac{|I(u)|}{|I|} - \mu (\mathbb{X}(V)) \nonumber \\
\hat{b}^{I}(\mathbb{X}(V),x,y) =~ \frac{\sum_{i} b(\mathbb{X}(V),i)}{|I(x) \cup I(y)|} \nonumber\\
b(\mathbb{X}(V),i) =~ \frac{|U(i)|}{|U|} - \mu (\mathbb{X}(V)) \nonumber 
\end{gather}$$
