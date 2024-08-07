## 실험환경 구성

### Anaconda 환경에서 모듈을 실행했습니다.
파이썬 버전은 3.12입니다.
```shell
# 환경 생성
conda create -n cfEnv python=3.12

# 환경실행
conda activate cfEnv

# 사용된 패키지들 설치
conda install -c conda-forge cryptography=42.0.5 openpyxl=3.1.2 openssl=3.3.1 paramiko=3.4.0 pymysql=1.0.2 sshtunnel=0.4.0 requests tqdm numpy scipy scikit-learn pandas matplotlib seaborn
```

## 나눔폰트
### 우분투에 설치

```bash
# 나눔 폰트들의 설치
apt-get install fonts-nanum*

# 폰트 캐시 삭제
fc-cache -fv

# matplotlib 경로 확인
python -c "import matplotlib; print(matplotlib.__file__)"

## 폰트 복사 cp ${FONTS_DIR_PATH} ${MATPLOTLIB_HOME}/mpl-data/fonts/ttf/
# cp /usr/share/fonts/truetype/nanum/Nanum* /opt/conda/envs/project/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/
cp ${FONTS_DIR_PATH} ${MATPLOTLIB_HOME}/mpl-data/fonts/ttf/
# MATPLOTLIB_HOME은 사용하는 conda환경에서의 matplotlib 경로입니다.

# matplotlib 폰트 캐시 삭제
rm -rf ~/.cache/matplotlib/*
```

### 맥에 설치

[참고](https://dev-adela.tistory.com/59)

### 파이썬에서 사용하기 

```python
# 전역 폰트 적용
import matplotlib.pyplot as plt
plt.rc('font', family='NanumGothicCoding')

# 마이너스가 깨질 경우
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False
```

[참고](https://velog.io/@redgreen/Linux-linux에서-Matplotlib-한글폰트-설정하기)
