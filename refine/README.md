
# 실험용 데이터 구성하기 (정답지 만들기)
요약: [예제코드](../ipynb/const_models_ex/dataset_creation.ipynb)
- 데이터 종류는 공개 데이터와 자사 데이터로 나뉘고, 종류 별로 처리를 달리하도록 모듈들이 구성됐습니다.
    - 자사 데이터: colley
    - 공개 데이터: movielens 
- 데이터 정제는 다음 순으로 처리합니다.
    1. 원시 데이터 구성
    2. 모집단 선정 (원시 데이터에서 실험에 사용할 데이터를 선별)
    3. 실험용 데이터 집합 구성 (교차 검증)
- 자사 데이터 기준으로 작성했습니다. (공개 데이터는 미사용하므로)

## 1. 원시데이터 구성
- [ColleyDatesItemsReductionRepository](../refine/colley/colley_dates_items_reduction_repository.py)를 사용해 원시데이터를 만드세요.
    - 데이터베이스(3090)에서 사용자, 상품, 게시글, 태그, 의사결정 데이터들을 복제합니다.
        - 상품과 게시글들을 통합해서, 통합항목을 구성합니다.
        - [데이터베이스 연결](../core/repo/base_repository.py), [데이터 복제 및 정제](../refine/colley/colley_dates_repository.py), [질의문](../refine/colley/colley_dates_queries.py)
            - [ShadowConnector](../core/repo/shadow_conn.py)
    - 분석기간을 정하고, 여기에 속하는 의사결정 내역들을 수집합니다.
        - 선별된 의사결정 내역에 속하는 사용자, 항목, 태그 등의 목록을 재구성합니다.
    - 데이터 규모 축소를 위해, 
        - 항목에 긍정한 의사결정 수가 임계 값 이상인 항목들을 선별합니다.
            - 긍정한 의사결정을 좋아요와 구매로 정의했기에,
            - `좋아요 한 사용자들의 수 + 구매한 사용자 수`를 사용합니다.
        - 아래의 예제에서는 임계 값을 160으로 했습니다.
            - 임계 값은 4분위수(quartile)로 큰 기준을 구하고, 업무용 장비 자원(램)에 할당 가능한 정도를 구할 때까지 높여가며 찾았습니다.
                - $|U|\times |T|^{2} \times 4(\text{float32}) < 16 \times 2^{20}$ GiB가 되는 임계 값 찾기
            - $좋아요+구매량 \geq 160$인 항목들만 실험용 데이터로 선별합니다.
            - 위 조건에서 탈락한 항목들의 의사결정 내역들도 실험용 데이터에서 삭제됩니다.
```python
from core import Machine
from refine import *

repo = ColleyDatesItemsReductionRepository(
    raw_data_dump_dir_path,
    db_src=Machine.E_MAC,
    begin_date_str="2023-11-01",
    emit_date_str="2023-12-31",
)
repo.items_reduction(positive_threshold=160)
```

<details>
<summary>매개변수 보기</summary>

raw_datas_dump_dir_path
- 데이터 베이스에서 복제된 원시 데이터가 저장될 `폴더경로`입니다.

db_src
- 접근할 데이터 베이스에 관한 열거자입니다. 업무용 노트북을 기본 값으로 합니다.

begin_date_str
- 의사결정 내역 수집의 시작날짜입니다.
- 해당일을 포함해 원시 데이터가 구성됩니다.

begin_date_str
- 의사결정 내역 수집의 종료날짜입니다.
- 해당일을 포함해 원시 데이터가 구성됩니다.

positive_threshold
- 항목에 긍정한 의사결정 수(좋아요, 구매 빈도의 합)의 임계 값입니다. 설정 값보다 작은 항목들은 여과됩니다.

</details>

## 2. 실험용 데이터 셋 구성 ($K$-fold cross-validation)
- [CrossValidationSplitter](../core/eval/splitter/cross_validation_splitter.py)로 실험에 사용할 교차검증 데이터 집합을 만드세요.
    - 위 절에서 처리(생성)한 데이터들의 폴더경로를 사용하면 됩니다. 

```python
from core import CrossValidationSplitter

splitter = CrossValidationSplitter(
    src_dir_path=raw_data_dir_path,
    dest_dir_path=kfold_dump_dir_path,
    fold_k=5,
    orded_timestamp=False,
)
splitter.split()
```

<details>
<summary>매개변수 보기</summary>

src_dir_path
- 원시 데이터가 저장된 `폴더경로`입니다.

dest_dir_path
- 교차검증 데이터 셋이 출력될 `폴더경로`입니다.

fold_k
- 원시 의사결정 내역들을 $K$개로 나눠서, 실험용 교차검증 집합들을 구성합니다.

orded_timestamp
- 원시 의사결정 내역들을 나누기 전, 의사결정된 시간을 기준으로 의사결정 내역들을 정렬합니다.
</details>
