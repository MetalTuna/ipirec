
## IPIRec(Rev 4) -- Model(Series 3)
### [IPIRecModelSeries3](./ipirec_model_series3.py)
$$\begin{gather}
\rho(x,y)=~ J(x,y)\Pr(x,y) \Pi_{n \in \forall sim} sim_{n}(x,y) \nonumber  \\
J(x,y)=~ \frac{|I(x)\cap I(y)|}{|I(x)\cup I(y)|} \nonumber  \\
\Pr(x,y)=~ \frac{|I(x)\cap I(y)|}{|I(x)|} \nonumber \\ 
sim_{UB}(x,y)=~ 
\frac{\sum_{u\in U(x)\cup U(y)}|NT(u) \cap x| |NT(u) \cap y|}
{\sqrt{\sum_{u}|NT(u) \cap x|\sum_{u}|NT(u) \cap y|}} \nonumber \\
sim_{UF}(x,y)=~ 
\frac{\sum_{u \in U(x)\cup U(y)}|I(u)\cap I(x)| |I(u)\cap I(y)|}
{\sqrt{\sum_{u}|I(u)\cap I(x)|^{2}\sum_{u}|I(u)\cap I(y)|^{2}}} \nonumber \\
sim_{IB}(x,y)=~ 
\frac{\sum_{i\in I(x)\cup I(y)}|T(i)\cap x| |T(i) \cap y|}
{\sqrt{\sum_{i}|T(i)\cap x|^{2}\sum_{i}|T(i)\cap y|^{2}}} \nonumber \\ 
sim_{IF}(x,y)=~ 
\frac{\sum_{i\in I(x)\cup I(y)}\tau(x) \tau(y)}
{\sqrt{\sum_{i}\tau(x)^{2}\sum_{i}\tau(y)^{2}}} \nonumber \\
\tau(t)=~ |I(t)|^{-1}\sum_{i \in I(t)}|T(i)| \nonumber \\ 
s(x,y) \leftarrow ~  \rho(x,y) + \bar{b}(x,y) \nonumber \\
\bar{b}(x,y) =~ \hat{b}^{U}(x,y) + \hat{b}^{I}(x,y) + \frac{b(x) + b(y)}{2} \nonumber \\
b(t) =~ \frac{|R(t)|}{|R|} - \mu(\mathbb{X}) \nonumber \\
\hat{b}^{U}(x,y) =~ \frac{\sum_{u \in U(x) \cup U(y)} b(u)}{|U(x) \cup U(y)|} \nonumber \\
b(u) =~ \frac{|I(u)|}{|I|} - \mu (\mathbb{X}) \nonumber \\
\hat{b}^{I}(x,y) =~ \frac{\sum_{i \in I(x) \cup I(y)} b(i)}{|I(x) \cup I(y)|} \nonumber \\
b(i) =~ \frac{|U(i)|}{|U|} - \mu (\mathbb{X}) \nonumber 
\end{gather}$$