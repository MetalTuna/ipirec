
# Decompositions
요약:
- $m$명의 사용자들의 $n$개 항목들에 대한 의사결정 행렬 $\mathbb{R}^{m \times n}$이 주어지면, 비음수 인수분해기로 $\mathbb{R}^{m \times n}$를 $d$차원으로 축소된 두 개의 행렬 $P$와 $Q$를 얻습니다.

    $$\mathbb{R}^{m \times n} \Rightarrow P^{m \times d}, Q^{n \times d} $$

    - $P^{m \times d}$: $d$차원으로 축소된 사용자들의 특성 행렬 (users latent factor)
    - $Q^{n \times d}$: $d$차원으로 축소된 항목들의 특성 행렬 (items latent factor)
    - 자사 실험용 데이터는 $m \geq n > d$의 구조를 갖습니다. 
- 사용자 $u$의 항목 $i$에 대한 선호정도를 다음과 같이 구합니다:

    $$\hat{r}(u,i) = P(u) \cdot Q(i)^{\text{T}}$$

    $$~~~~~~~~~~~~~=\sum_{f = 1}^{d} p_{u,f} \times q_{i,f}$$

핵심어
- hill climbing: gradient descent로 latent factors를 feedback
- online learning: 학습용 모집단 선별(samples) 미구현
- heuristic approach: objective function, momentum 미구현

## 모듈구성

인수분해기와 모델 피드백 기능분리 목적으로, 인수분해기는 `factorizer`, 모델 피드백은 `optimizer`로 다시 나눴습니다.

### factorizer
- [DecompositionModel(BaseModel)](factorizer/decomposition_model.py): 의사결정 행렬을 만들고, 인수분해기로 의사결정 행렬을 분해하는 기능을 추상화하는 클래스입니다.

- [NMFDecompositionModel(DecompositionModel)](factorizer/nmf_model.py): `scipy`에 구현된 비음수 행렬 분해기로 의사결정 행렬을 분해합니다.

### optimizer

- [DecompositionsEstimator(BaseTrain)](optimizer/decompositions_estimator.py): 두 개의 특성행렬을 사용해 선호정도를 추정하고, 오차를 보정하는 기능이 구현된 클래스입니다.
    - 다음과 같이 예측 오차를 구하고, 오차를 사용해 특성 값들을 보정합니다:

        $$\epsilon(u,i) = r(u,i) - \hat{r}(u,i)$$

        $$p_{u} = p_{u} + \lambda (\epsilon(u,i) \cdot q_{i} - \gamma \cdot p_{u})$$

        $$q_{i} = q_{i} + \lambda (\epsilon(u,i) \cdot p_{u} - \gamma \cdot q_{i})$$

    - 목적함수는 미구현됐지만, 구상안은 다음과 같습니다:

        $$\arg_{P, Q} \min (\frac{1}{N\times |\mathbb{X}|}\sum_{(u,i) \in \mathbb{X}} (r(u,i)-\hat{r}(u,i))^{N})^{\frac{1}{N}} + \gamma (\left\lVert P \right\rVert _{N}+\left\lVert Q \right\rVert _{N})$$

        여기에 사용된 $\left\lVert * \right\rVert_{N}$는 $L_{N}$-Norm(frobenious norm)입니다($N=[1,2]$로 설계).

- BaselineEstimator(DecompositionsEstimator): 미사용
    - DecompositionsEstimator를 완성하면 예측성능의 극대화를 목적으로 설계 중이었던 모듈임(특이 값 분해로 확장할 목적).

- 모델구성: 콜리 실험용 데이터 셋 분석을 예로 서술합니다.
    - 사용된 모듈들
        - DataSet: [ColleyDataSet](../colley/dataset/colley_dataset.py)
        - Model: [NMFDecompositionModel](factorizer/nmf_model.py)
        - Estimator: [DecompositionsEstimator](optimizer/decompositions_estimator.py)
        - Recommender: [ScoreBasedRecommender](../core/recommenders/score_based_recommender.py)
    - 예제:
        - 자세한 구현은 [코드](../experiments/related/nmf.py)를 참고하세요.
        ```python
        _dataset = ColleyDataSet(DATA_SET_HOME_PATH)
        _dataset.load_kfold_train_set(KFOLD_NO)

        _model = NMFDecompositionModel(
            _dataset,
            FACTORS_DIM,
            FACTORIZER_ITERS,
            )
        _model.analysis()

        _estimator = DecompositionsEstimator(
            _model,
            FEEDBACK_ITERS,
            LAMBDA,
            GAMMA,
            FROB_NORM,
            )
        _estimator.train()
        
        _recommender = ScoreBasedRecommender(_estimator)
        _recommender._prediction_()
        ```
