# Experiments

### Modules

|Methods|Models|Runners|Evaluations|
|:--:|:--:|:--:|:--:|
|Base|[IBCF](./related/ibcf.py)|[sh](./related/run_ibcf.sh)|[IRMetrics](./related/ibcf_results_summary.py)|
|Base|[NMF](./related/nmf.py)|[sh](./related/run_nmf.sh)|[IRMetrics](./related/nmf_results_summary.py)|
|Our|[Rev1](./our/rev1.py)|[sh](./our/run_rev1.sh)|[IRMetrics](./our/rev1_results_summary.py)|
|Our|[Rev2](./our/rev2.py)|[sh](./our/run_rev2.sh)|[IRMetrics](./our/rev2_results_summary.py)|
|Our|[Rev4](./our/rev4.py)|[sh](./our/run_rev4.sh)|[IRMetrics](./our/results_summary.py)|
|Our|[Rev4Ver1](./our/rev4v1.py)|[sh](./our/run_rev4ver1.sh)| N/A |

### Models parameters description

<details>
<summary>IBCF</summary>

- distance: Pearson correlation coefficient
- estimation: Adjusted weighted sum
</details>

<details>
<summary>NMF</summary>

- factorizer: $f=200, iter=200$
- estimator: $\lambda = 10^{-4}, \gamma = 10^{-5}, iter=150, L_{1}$-Norm
    - Seq.: $\tau^{\text{main}}=[V,L,P]$.
</details>

<details>
<summary>Rev1</summary>

- model: $NT(u)=10, \theta_{I}=10, \nu=0$
- estimator: $L_{1}$-Norm
    - Scores: $iter=10, \lambda=10^{-2}, \gamma=10^{-2}$
    - Weights: $iter=10, \lambda=10^{-2}, \gamma=10^{-2}$
    - Seq.: $\tau^{\text{main}}=[V,L,P]$.
</details>

<details>
<summary>Rev2</summary>

- model: $NT(u)=5, \theta_{I}=4, \nu=0$
- estimator: $L_{1}$-Norm
    - Scores: $iter=10, \lambda=10^{-2}, \gamma=10^{-4}$
    - Weights: $iter=5, \lambda=10^{-3}, \gamma=10^{0}$
    - Seq.: $\tau^{\text{main}}=[V,L,P]$.
</details>

<details>
<summary>Rev4</summary>

- estimator: $L_{1}$-Norm
    - Scores: $\lambda=10^{-2}, \gamma=10^{-4}$
    - Weights: $iter=5, \lambda=10^{-3}, \gamma=10^{0}$
    - Seq.: $\tau^{\text{main}}=[V,L,P], \tau^{\text{post}}=[V]$.
</details>

<details>
<summary>Rev4Ver1</summary>

- estimator: $L_{1}$-Norm
    - Scores: $\lambda=10^{-2}, \gamma=10^{-4}$
    - Weights: $iter=5, \lambda=10^{-3}, \gamma=10^{0}$
    - Seq.: $\tau^{\text{main}}=[V,L,P], \tau^{\text{post}}=[V]$.
</details>

### Results comparision
Benchmark models: IBCF, NMF, Rev1, Rev4Ver1.

<details>
<summary>Means</summary>

- $f_{1}$-scores

    ![AvgF1](../assets/figs/AvgF1.svg)

- No. of hits

    ![AvgHits](../assets/figs/AvgHits.svg)

</details>

<details>
<summary> Top-N</summary>

- $f_{1}$-scores

    ![LikesF1](../assets/figs/TopN_f1_Likes.svg)

    ![PurchasesF1](../assets/figs/TopN_f1_Purchases.svg)

- No. of hits

    ![LikesHits](../assets/figs/TopN_Hits_Likes.svg)

    ![PurchasesHits](../assets/figs/TopN_Hits_Purchases.svg)

</details>

<details>
<summary>Tags distributions</summary>

![TagsFreqDist](../assets/figs/TagsFreqDist.svg)
</details>


<details>
<summary>Tags scores</summary>

![rmseAll](../assets/figs/RMSE_All.svg)

![rmseHits](../assets/figs/RMSE_Hits.svg)

</details>