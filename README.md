# Cell Arena

강화학습(DQN, Dueling DQN, PPO) 실습용 세포 대전 환경.
`agent.py` 하나에 에이전트와 학습 루프를 구현해 학습하고, 서로 대결함.

## 시작하기

### 0. 설치

`uv` 가 없으면 운영체제에 맞게 한 번 설치.

macOS (Homebrew):

```bash
brew install uv
```

Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

설치 후 터미널을 새로 열고 `uv --version` 확인.
Git 도 설치되어 있어야 함 (`git --version` 으로 확인).
이 저장소 폴더 (`requirements.txt` 가 있는 위치)에서 Python 3.11 가상환경 생성.
해당 Python 이 없으면 uv 가 자동으로 내려받음.

```bash
uv venv --python 3.12
```

가상환경 활성화 (터미널을 새로 열 때마다 실행):

```bash
# macOS / Linux (bash, zsh)
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

활성화한 터미널에서 의존성 설치와 점검.
Linux + NVIDIA GPU 는 설치 전에 `requirements.txt` 의 CUDA 주석 참고.

```bash
uv pip install -r requirements.txt
cell-arena-doctor  # 설치 점검. 마지막 줄이 "모두 정상" 이면 됨
wandb login
```

이후 모든 `python`, `cell-arena-*`, `wandb` 명령은 가상환경을 활성화한 터미널에서 실행.
설정 파일을 바꿔도 다시 설치할 필요는 없고, `requirements.txt` 를 바꾸면 설치 명령을 다시 실행.

### 1. 게임 직접 플레이

```bash
cell-arena-play                           # 기본 봇들과 플레이
cell-arena-play --opponents gold diamond  # 상대 지정
cell-arena-check dqn_agent.py             # 예제 에이전트 점검. [통과] 가 나오면 됨
```

창이 뜨고 조작이 되는지, 밥을 먹으면 커지고 큰 세포에 먹히면 리스폰되는지 확인.

### 2. 학습 돌려보고 로그 확인

`config.yaml` 을 `test.yaml` 로 복사해 아래 두 줄만 바꾸고 예제로 짧게 학습.

```yaml
total_samples: 100_000
learning_starts: 20_000
```

```bash
uv run python dqn_agent.py --config test.yaml    # 끝나면 dqn.pt 생성
uv run python dueling_dqn_agent.py --config test.yaml  # 끝나면 dueling_dqn.pt 생성
uv run python ppo_agent.py --config test.yaml
```

- 터미널 첫 줄에 `[cell-arena] env ...` 가 찍히면 env 정상
- wandb 대시보드(`https://wandb.ai` 의 `cell-arena` 프로젝트)에 `train/*` 그래프가 10,000 샘플마다 찍히는지 확인
- 인터넷이 없으면 `WANDB_MODE=offline uv run python dqn_agent.py --config test.yaml`, 나중에 `wandb sync wandb/offline-run-*`

### 3. 내 컴퓨터에 맞게 설정 조율

램 용량 확인 (macOS: 활동 모니터 > 메모리, Windows: 작업 관리자 > 성능 > 메모리, Linux: `free -h`).
학습이 램의 절반 정도만 쓰게 맞추는 걸 권장.

메모리를 가장 많이 쓰는 건 저장해 두는 관측 (`frame_stack: 4`, `resolution: 64` 기준).

| 대상 | 메모리 | 조절하는 키 |
|---|---|---|
| DQN 리플레이 버퍼 | `buffer_size` x 200KB | `buffer_size` |
| PPO rollout | `n_steps` x `num_envs` x 100KB | `n_steps`, `num_envs` |

| 램 | DQN `buffer_size` | PPO `num_envs` x `n_steps` |
|---|---|---|
| 8GB | 10_000 (약 2.0GB) | 32 x 128 (약 0.4GB) |
| 16GB | 30_000 (약 5.9GB) | 64 x 128 (약 0.8GB) |
| 32GB 이상 | 60_000 (약 11.8GB) | 64 x 256 (약 1.6GB) |

- `frame_stack` 이나 `resolution` 을 키우면 메모리도 비례해서 증가 (`frame_stack` x `resolution`^2)
- 학습이 너무 느리면 `num_envs` 를 줄이거나 `cell-arena-doctor` 의 처리량(샘플/s) 확인
- 학습 중 램이 가득 차면 (스왑 발생, 컴퓨터가 멈칫함) 위 키들을 줄이고 다시 실행

### 4. 직접 설계

`agent.py` 를 채우거나 예제 하나를 `agent.py` 로 복사해 시작. 아래 "할 일" 참고.
관측 전처리, 모델, 보상, 학습 루프를 바꿔 가며 wandb 로 비교.

## 할 일

`agent.py` 와 `config.yaml` 만 수정.

- `obs_spec` / `action_spec`: 관측(`image` / `state`)과 액션(`discrete` / `continuous`) 형태
- `setup`: 모델 (DQN / Dueling DQN / PPO)
- `preprocess`: 관측 -> 신경망 입력
- `policy`: 행동 선택
- `reward`: 보상
- `train`: 학습 루프 (버퍼, 업데이트, wandb 로그)

`__init__`, `act`, `save`, `load` 는 수정 불가. `load` 는 저장된 설정으로 `setup` 을 다시 부르므로 모델 구조는 `self.cfg` 만으로 정해져야 함.
`config.yaml` 에는 상대 봇 목록과 하이퍼파라미터를 적음.

```bash
python agent.py                   # 학습 -> my_agent.pt
python agent.py --config dqn.yaml # 다른 설정 파일로
cell-arena-check agent.py         # 제출 전 점검
cell-arena-play                   # 직접 플레이
cell-arena-play --opponents gold agent.py   # 상대 지정 (봇 이름 또는 에이전트 파일)
```

제출: 학습 전에 `name` 과 `weights` 를 `"<이름>"`, `"<이름>.pt"` 로 변경. 학습이 끝나면 `agent.py` 를 `<이름>.py` 로 바꿔 `<이름>.pt` 와 함께 제출. 대결장은 `weights` 에 적힌 파일을 불러옴. 다른 .py 는 import 불가.

### DQN -> Dueling DQN 실습 순서

`agent.py` 는 빈칸을 직접 채우는 스켈레톤. `setup` 과 `policy` 의 `NotImplementedError` 는 구현 후 제거.
먼저 DQN 을 완성하고, 그다음 Q 네트워크를 Dueling 구조로 바꿈.
학습 루프의 TODO 1~8 은 두 알고리즘에서 동일하게 사용.

| 단계 | 구현할 내용 | 중간 확인 |
|---|---|---|
| 모델·행동 | `setup`, `policy` 구현 | Q 출력 `(B, 18)`, 행동은 0~17 정수 배열 `(B,)`. `explore=False` 에서는 탐험하지 않음 |
| TODO 1 | 타깃 네트워크, 옵티마이저, 버퍼, 스케줄 준비 | 타깃은 학습 모델과 별도 객체이며 초기 가중치는 같음 |
| TODO 2~3 | 탐험률 갱신, 현재 입력 확보, 다음 입력 구성, 전이 저장 | 전처리는 스텝당 한 번. 다음 입력은 `final_obs` 와 `frames.peek` 로 구성 |
| TODO 4~5 | 미니배치의 Q(s, a) 선택, TD 타깃 계산 | 둘 다 `(batch_size,)`. 타깃에는 기울기가 없고 `terminated` 만 부트스트랩을 막음 |
| TODO 6~7 | loss 계산, 역전파, 파라미터 갱신, 타깃 동기화 | 학습 모델은 업데이트되고 타깃은 정해진 샘플 간격에만 복사됨 |
| TODO 8 | loss 와 탐험률 로그 추가 | `train/loss`, `train/epsilon` 을 기존 로그와 함께 기록 |
| Dueling 전환 | 공통 특징에서 V 와 A 를 따로 출력해 결합 | V 는 `(B, 1)`, A 는 `(B, 18)`. 행동 축 평균을 빼서 Q 는 `(B, 18)` 유지 |

막히면 `dqn_agent.py` 의 `ReplayBuffer`, `update`, `train` 을 해당 단계별로 참고.
Dueling 전환은 `dueling_dqn_agent.py` 의 `DuelingQNet` 과 비교.
두 모델의 체크포인트는 구조가 다르므로 전환 전 DQN 파일을 별도 보관하고, Dueling 은 다른 `weights` 이름으로 새로 학습.

### 실습 완료 기준

먼저 위의 `test.yaml` (10만 샘플, `learning_starts: 20_000`) 로 짧게 확인.
버퍼 크기와 병렬 환경 수는 메모리 조율 표에 맞게 설정.
아래 명령은 스켈레톤을 구현한 뒤 실행하며, `name` / `weights` 를 바꿨다면 저장 파일명도 그에 맞춰 확인.

```bash
uv run python agent.py --config test.yaml
cell-arena-check agent.py
cell-arena-play --opponents gold agent.py
```

- [ ] **행동 규격:** `cell-arena-check agent.py` 에서 `[통과]` 가 나옴.
- [ ] **학습 업데이트:** `learning_starts` 이후 `train/loss` 가 NaN/Inf 가 아닌 값으로 기록됨.
  실제 업데이트 전후 학습 모델의 파라미터를 복사해 비교하여, 값이 바뀌는 것도 확인.
  loss 가 계속 감소해야 하는 것은 아니며, 유한한 loss 만으로 업데이트 성공을 판정하지 않음.
- [ ] **탐험률:** `train/epsilon` 이 설정한 시작값에서 하한까지 스케줄에 따라 감소함.
  짧게 실행하면 하한까지 도달하지 않아도 됨.
- [ ] **에피소드 로그:** 최소 한 에피소드가 종료된 뒤 `train/episode_return` 이 유한한 값으로 기록됨.
  첫 에피소드가 끝나기 전의 NaN 은 정상. 로그는 10,000 샘플마다 기록됨.
- [ ] **저장:** 학습이 정상 종료되고 `weights` 에 지정한 파일(기본 `my_agent.pt`)이 생성됨.
- [ ] **재로딩·행동:** 학습 종료 후 위 플레이 명령을 별도 프로세스로 실행.
  저장한 모델이 로드되고, shape 오류 없이 에이전트가 게임에 참여해 행동하는지 확인.
- [ ] **Dueling:** 네트워크를 바꾼 뒤 같은 기준을 다시 통과하고, V/A 출력 크기와 결합식을 설명할 수 있음.

여기까지는 **구현·실행 완료** 기준이며, 봇을 이기는 것이 필수는 아님.
**성능 향상**을 주장하려면 학습 전후 모델을 같은 상대·평가 조건에서 `explore=False` 로 여러 번 평가해 비교.
학습 중 보상이나 한 번의 대결 결과만으로 판단하지 않음.
예제의 기본 탐험 스케줄은 10만 샘플 시점에도 무작위 행동 확률이 약 90.5% 이므로,
짧은 학습 중의 행동과 탐험을 끈 평가 행동은 구분해야 함.

## 예제

`agent.py` 를 채운 기본 구현. 관측은 모두 이미지 4 프레임.

| 파일 | 알고리즘 | 액션 |
|---|---|---|
| `dqn_agent.py` | DQN | `discrete` |
| `dueling_dqn_agent.py` | Dueling DQN | `discrete` |
| `ppo_agent.py` | PPO | `continuous` |

하이퍼파라미터 기본값은 각 파일의 `cfg.get(...)` 에 있음. `config.yaml` 에 적으면 덮어씀.

## 게임 규칙

맵은 100 x 100 토러스. `s` = 세포 크기.

### 객체

| 객체 | 설명 |
|---|---|
| 세포 | 플레이어. 크기 100 에서 시작, 직경 `2 * sqrt(s/100)` |
| 밥 | 맵에 300개. 먹으면 밥 크기의 절반만큼 커짐. 먹히면 다른 곳에 다시 생김 |
| 세포밥 | 세포가 돌진할 때 흘리는 밥. 먹으면 크기의 절반만큼 커짐. 다시 생기지 않음 |
| 블랙홀 | 15개. 크기 500 초과인 세포가 닿으면 -250. 발동하면 다른 곳에 다시 생김 |
| 화이트홀 | 15개. 크기 500 미만인 세포가 닿으면 +100. 발동하면 다른 곳에 다시 생김 |

### 조작

| 입력 | 동작 |
|---|---|
| 이동 | 8방향 또는 정지 (`continuous` 액션은 임의 각도). 클수록 느리고 방향 전환이 둔함 |
| 돌진 | 이동 중에만 가능. 클수록 빠르지만 4스텝마다 크기의 `1% * sqrt(s/100)` 를 세포밥으로 흘림. 크기 100 이하면 불가 |
| 직접 플레이 | `W` `A` `S` `D` 이동, `SPACE` 돌진, `TAB` 패널 세포 변경, `R` 리셋, `ESC` 종료 |

### 규칙

| 규칙 | 내용 |
|---|---|
| 포식 | 세포끼리 경계면이 닿으면 큰 쪽이 작은 쪽을 통째로 먹음 (같은 크기면 무작위) |
| 사망 | 먹히거나, 블랙홀을 한 번에 여러 개 밟아 크기가 0 이하가 될 때 |
| 시야 | 한 변 `2 * min(8 * sqrt(s/100), 25)` 인 정사각형. 그 밖은 보이지 않음 |
| 승리 | 크기 4000 |
| 학습 중 | 죽으면 크기 100 으로 바로 리스폰하고 에피소드는 계속됨 |
| 대결 | 먹히면 탈락. 누군가 4000 에 도달하거나 한 명만 남으면 끝 |

## 관측

첫 차원은 배치 B (학습 = `num_envs`, 대결 = 1). 좌표는 내 세포 기준, y 는 아래가 +.

| 필드 | 형태 | 내용 |
|---|---|---|
| `self_state` | `(B, 5)` | `[크기, x, y, v_x, v_y]` |
| `objects` | `(B, M, 8)` | state 모드. 시야 안 객체 가까운 순 `[dx, dy, 크기, is_food, is_cell_food, is_black_hole, is_white_hole, is_cell]` |
| `mask` | `(B, M)` | state 모드. 실제 객체면 True, 나머지는 0 패딩 |
| `image` | `(B, 6, R, R)` | image 모드. 채널 `[밥, 세포밥, 블랙홀, 화이트홀, 다른 세포, 나]`, 값 0/1 |

- `ObsSpec(mode="image", resolution=64)` (기본) 또는 `ObsSpec(mode="state", max_objects=32)`
- 이미지는 크기와 상관없이 R x R 이라 클수록 넓고 거칠게 보임
- 밥과 세포밥은 따로 구분됨 (`is_food` / `is_cell_food`, 이미지 채널 0 / 1). 순서는 `cell_arena.OBJECT_FEATURES`, `cell_arena.IMAGE_CHANNELS`
- 다른 세포의 속도는 주지 않음

## 액션

| 모드 | 형태 | 내용 |
|---|---|---|
| `discrete` | `(B,)` 정수 0~17 | 0~8 = 정지, 상, 하, 좌, 우, 좌상, 우상, 좌하, 우하 / 9~17 = 같은 순서 + 돌진 |
| `continuous` | `(B, 3)` | `[theta, move, dash]`. theta 는 오른쪽 0, 아래 +pi/2 (라디안). move, dash 는 0.5 초과면 켜짐 |

## 사건 (`events`)

환경은 보상을 주지 않고 내 세포에게 일어난 사건만 줌. 각 `(B,)`.

| 필드 | 내용 |
|---|---|
| `size_before`, `size_after` | 스텝 전후 크기 |
| `food_mass` | 밥, 세포밥으로 얻은 질량 |
| `white_hole`, `black_hole` | 발동 횟수 |
| `dash_cost` | 돌진으로 흘린 질량 |
| `kills`, `kill_mass` | 잡아먹은 세포 수와 얻은 질량 |
| `died`, `won` | 죽었는지, 4000 에 도달했는지 |
| `t` | 에피소드 경과 스텝 |

## 학습 env

`env.step(action)` 은 `obs, final_obs, events, terminated, truncated` 를 반환.

- `terminated`: 누군가 4000 도달. 부트스트랩하지 않음
- `truncated`: `max_steps` 도달. `final_obs` 로 부트스트랩함
- 끝난 env 는 자동 리셋됨. `obs` 는 새 에피소드의 첫 관측, `final_obs` 는 리셋 전 마지막 관측
- 죽어도 에피소드는 끝나지 않음 (`events.died`)
