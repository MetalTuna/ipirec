[수정일] 2024.07. 24
분석모듈 연구개발시 사용된 모듈집합입니다.

# IP상품 특징분석: 태그 상관관계를 사용한 라이센스 상품추천 (IPIRec, IP items recommendation using tags correlation)
- IP상품의 태그 상관관계에 근거해서, 상품 선호도를 예측하는 추천 모델입니다.

# 비교모델
1. 항목기반 협업 필터링 (IBCF, Item-based collaborative filtering)
- 항목 간의 유사도에 근거해서, 상품 선호도를 예측하는 추천 모델입니다.
2. 행렬분해 추천 (MFRec, Matrix factorizaiton based recommendation)
- 사용자와 항목에 대한 의사결정 행렬을 몇 개의 작은 차원을 갖는 행렬로 특성을 표현하고, 상품 선호도를 예측하는 추천 모델입니다.

# 우리의 모델
태그 상관관계를 사용한 라이센스 상품추천 (IPIRec, IP items recommendation using tags correlation)

# 실험용 데이터 셋
1. 공개 데이터 (MovieLens) - 아래의 사유로 추천 시스템 연구분야에서 관례적으로 사용되는 공개 데이터 셋을 분석합니다.
- 자사 데이터 셋의 공정성 문제 제기에 대한 대비
- 우리 분석모델의 공정성/적합성 평가 문제 제기에 대한 대비

2. 자사 데이터 (Colley)
- 콜리 앱 사용자들의 의사결정 데이터
- 상품이나 게시글의 열람, 좋아요, 구매여부 등의 정보들을 의사결정으로 정의합니다.
