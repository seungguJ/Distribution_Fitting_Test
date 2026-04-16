# Distribution Fitting Test

## 개요

이 프로젝트는 `0`부터 `N`까지의 정수 범위에서 생성한 랜덤 데이터셋으로부터
경험적 누적분포(CDF)를 만들고, 이를 아주 작은 ML 회귀 모델로 근사하는 실험입니다.

실험의 초점은 다음과 같습니다.

- 데이터가 학습하기 좋게 정리되어 있다는 가정을 두지 않음
- 낮은 값에 몰린 데이터, 높은 값에 몰린 데이터, 양 끝에 몰린 데이터를 모두 테스트
- 평균과 분산을 사용자가 조절하면서 분포 특성을 바꿔볼 수 있음
- 단순한 모델이 불균형한 랜덤 데이터의 CDF를 얼마나 따라가는지 확인

## 실험 흐름

1. `config.yaml`에서 `N`, `sample_size`, `mean`, `variance` 등을 설정
2. 설정에 따라 `0..N` 범위의 랜덤 데이터를 생성
3. 생성된 데이터로 경험적 CDF를 계산
4. `torch` 기반 회귀 모델을 학습
5. 실제 CDF와 예측 CDF를 비교하고 결과를 저장

## Config 예시

```yaml
N: 100
sample_size: 5000
mean: 30
variance: 200
seed: 42
distribution_mode: mixed
train_ratio: 0.8
hidden_size: 12
epochs: 4000
learning_rate: 0.08
```

주요 설정:

| 항목 | 설명 |
| --- | --- |
| `N` | 데이터 최대값. 생성 범위는 `0..N` |
| `sample_size` | 샘플 수 |
| `mean` | 분포 중심 조절값 |
| `variance` | 분포 퍼짐 조절값 |
| `seed` | 재현 가능한 실험용 시드 |
| `distribution_mode` | `lognormal`, `mixed`, `low_biased`, `high_biased`, `wide_spread`, `edge_focused`, `noisy_random` |
| `train_ratio` | CDF 포인트 학습 비율 |
| `hidden_size` | `torch` 모델의 hidden width |
| `epochs` | 학습 반복 수 |
| `learning_rate` | SGD 학습률 |

현재 예측기는 `torch` 기반 모델입니다.

- hidden layer는 `Softplus`를 사용
- 마지막 출력은 `sigmoid` 기반으로 만들어 `0 <= y < 1` 범위를 유지
- 입력이 항상 `0` 이상이라는 조건을 이용해 `raw_output = x * positive_scale` 구조를 사용
- 따라서 입력이 `0`일 때 출력은 정확히 `0`

파라미터 수는 모델 width에 따라 달라지며, 실행 시 자동으로 계산해 메트릭에 기록합니다.

## 분포 모드

- `lognormal`: 오른쪽 꼬리가 긴 로그노말 형태. CDF가 강한 곡선 형태를 띔
- `low_biased`: 낮은 값 주변에 상대적으로 몰림
- `high_biased`: 높은 값 주변에 상대적으로 몰림
- `wide_spread`: 넓게 퍼지도록 생성
- `edge_focused`: 작은 값과 큰 값 양쪽 끝에 몰림
- `noisy_random`: 구조가 약한 랜덤 분포
- `mixed`: 여러 모드를 섞어서 더 불규칙하게 생성

`mean`과 `variance`는 각 모드의 중심과 퍼짐을 조절하는 힌트로 사용됩니다.
랜덤 생성이기 때문에 실제 샘플 평균과 분산은 설정값과 정확히 일치하지 않을 수 있습니다.

## 실행 방법

의존성 설치:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

실행:

```bash
python3 main.py
```

다른 설정 파일을 쓰고 싶다면:

```bash
python3 main.py --config config.yaml
```

## 출력 결과

실행이 끝나면 `outputs/` 아래에 다음 파일이 생성됩니다.

- `outputs/metrics/latest_metrics.json`: 오차 지표와 데이터 요약
- `outputs/plots/latest_cdf.svg`: 실제 CDF와 예측 CDF 비교 그래프
- `outputs/data/latest_samples.csv`: 생성된 샘플 일부
- `outputs/data/latest_cdf.csv`: 실제/예측 CDF 값

## 평가 지표

- `MSE`: 실제 CDF와 예측 CDF의 평균제곱오차
- `MAE`: 절대오차 평균
- `R^2`: 회귀 적합도
- `monotonic_violations`: 예측 CDF가 감소한 구간 수

## 파일 구조

```text
Distribution_Fitting_Test/
├─ README.md
├─ config.yaml
├─ main.py
├─ data_generator.py
├─ cdf_builder.py
├─ model.py
├─ evaluator.py
└─ outputs/
```
