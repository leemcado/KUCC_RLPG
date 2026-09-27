# Cell Arena

강화학습(DQN, Dueling DQN, PPO) 실습용 세포 대전 환경.
`agent.py` 하나에 에이전트와 학습 루프를 구현해 학습하고, 서로 대결함.

## 시작하기

### 0. 설치

```bash
pip install -r requirements.txt   # Linux + NVIDIA GPU 는 requirements.txt 주석 참고
cell-arena-doctor                 # 설치 점검. 마지막 줄이 "모두 정상" 이면 됨
wandb login
```

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
python dqn_agent.py --config test.yaml    # 끝나면 dqn.pt 생성
python ppo_agent.py --config test.yaml
```

- 터미널 첫 줄에 `[cell-arena] env ...` 가 찍히면 env 정상
- wandb 대시보드(`https://wandb.ai` 의 `cell-arena` 프로젝트)에 `train/*` 그래프가 10,000 샘플마다 찍히는지 확인
- 인터넷이 없으면 `WANDB_MODE=offline python dqn_agent.py --config test.yaml`, 나중에 `wandb sync wandb/offline-run-*`

### 3. 내 컴퓨터에 맞게 설정 조율

램 용량 확인 (macOS: 활동 모니터 > 메모리, Windows: 작업 관리자 > 성능 > 메모리, Linux: `free -h`).
학습이 램의 절반 정도만 쓰게 맞추는 걸 권장.

메모리를 가장 많이 쓰는 건 저장해 두는 관측 (`frame_stack: 4`, `resolution: 64` 기준).

| 대상 | 메모리 | 조절하는 키 |
|---|---|---|
| DQN 리플레이 버퍼 | `buffer_size` x 160KB | `buffer_size` |
| PPO rollout | `n_steps` x `num_envs` x 80KB | `n_steps`, `num_envs` |

| 램 | DQN `buffer_size` | PPO `num_envs` x `n_steps` |
|---|---|---|
| 8GB | 10_000 (약 1.6GB) | 32 x 128 (약 0.3GB) |
| 16GB | 30_000 (약 4.9GB) | 64 x 128 (약 0.7GB) |
| 32GB 이상 | 60_000 (약 9.8GB) | 64 x 256 (약 1.3GB) |

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
| `objects` | `(B, M, 7)` | state 모드. 시야 안 객체 가까운 순 `[dx, dy, 크기, is_food, is_black_hole, is_white_hole, is_cell]` |
| `mask` | `(B, M)` | state 모드. 실제 객체면 True, 나머지는 0 패딩 |
| `image` | `(B, 5, R, R)` | image 모드. 채널 `[밥, 블랙홀, 화이트홀, 다른 세포, 나]`, 값 0/1 |

- `ObsSpec(mode="image", resolution=64)` (기본) 또는 `ObsSpec(mode="state", max_objects=32)`
- 이미지는 크기와 상관없이 R x R 이라 클수록 넓고 거칠게 보임
- 세포밥은 밥과 같은 `food` 로 보임. 다른 세포의 속도는 주지 않음

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
