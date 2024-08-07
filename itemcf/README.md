
# IBCF
- 모듈구성:

    ```mermaid
    graph TD;
        BaseModel-->BaseDistanceModel;
        BaseDistanceModel-->Pearson;
        BaseEstimator-->WeightedSum;
        BaseEstimator-->AdjustedWeightedSum;
    ```

    - [BaseDistanceModel(BaseModel)](./base_distance_model.py): 항목들 간의 유사도(거리) 계산을 추상화할 목적으로 기능이 재정의 된 추상클래스입니다.
    - [Pearson(BaseDistanceModel)](./pearson.py): 항목들 간의 거리를 피어슨 상관계수로 구합니다.
        - 두 항목에 대한 피어슨 상관계수는 다음과 같습니다:
      
        $$sim(i,j) = \frac{\sum_{u \in U(i) \cup U(j)} (r(u,i) - \bar{r}(i))(r(u,j) - \bar{r}(j))}{\sqrt{\sum_{u}(r(u,i) - \bar{r}(i))^{2}\sum_{u}(r(u,j) - \bar{r}(j))^{2}}}$$
      
        $$\bar{r}(i) = \frac{|U(i)|}{|U|}$$

        - $r(u,i)$: 사용자 $u$의 항목 $i$에 대한 의사결정 여부 ($r(u,i) = [0, 1]$)
        - $|U|$: 학습데이터에 속한 모든 사용자들 집합의 길이(사용자들의 수)
        - $|U(i)|$: 항목 $i$에 의사결정한 사용자들 집합의 길이

    - [WeightedSum(BaseEstimator)](./weighted_sum.py): 유사도를 가중치로 하는 가중치 합으로 선호정도를 예측합니다.
        - 사용자 $u$의 항목 $i$에 대한 선호정도를 다음과 같이 구합니다:

        $$\hat{r}(u,i) = \frac{\sum_{j \in I(u)} r(u,j) \times sim(i,j)}{\sum_{j \in I(u)} |sim(i,j)|}$$

        - 수식에 사용된 변수 $I(u)$는 사용자 $u$가 의사결정한 항목들의 집합, $sim(i,j)$는 항목 $i$와 $j$간의 유사도, $r(u,j)$는 사용자 $u$의 항목 $i$에 대한 의사결정 여부(혹은 정도)입니다.
    - [AdjustedWeightedSum(BaseEstimator)](./adjusted_weighted_sum.py)
        - 보정된 가중치 합은 선형회기 특성을 가할 목적으로, 가중치 합을 변형해 계산합니다.

        $$\hat{r}(u,i) = \bar{r}(i)+ \frac{\sum_{j \in I(u)} (r(u,j)-\bar{r}(j)) \times sim(i,j)}{\sum_{j \in I(u)} |sim(i,j)|}$$
        - $\bar{r}(i)$는 항목 $i$의 평균입니다. 자사 분석모델은 의사결정 여부를 구할 목적으로 설계했기에, 평균 의사결정 빈도 수가 사용됩니다.

- 모델구성: 콜리 실험용 데이터 셋 분석을 예로 서술합니다.
    - 사용된 모듈들
        - DataSet: [ColleyDataSet](../colley/dataset/colley_dataset.py)
        - Model: [Pearson](./pearson.py)
        - Estimator: [AdjustedWeightedSum](./adjusted_weighted_sum.py)
        - Recommender: [ScoreBasedRecommender](../core/recommenders/score_based_recommender.py)
    - 예제:
        - 자세한 구현은 [코드](../experiments/related/ibcf.py)를 참고하세요.
        ```python
        _dataset = ColleyDataSet(DATA_SET_HOME_PATH)
        _dataset.load_kfold_train_set(KFOLD_NO)

        _model = Pearson(_dataset)
        _model.analysis()

        _estimator = AdjustedWeightedSum(_model)
        
        _recommender = ScoreBasedRecommender(_estimator)
        _recommender._prediction_()
        ```
        
