# Cell Arena

강화학습(DQN · Dueling DQN · PPO) 실습용 세포 대전 환경.
`agent.py` 하나에 에이전트와 학습 루프를 구현해 학습하고, 서로 대결한다.

## 설치

```bash
pip install -r requirements.txt   # Linux + NVIDIA GPU 는 requirements.txt 주석 참고
cell-arena-doctor                 # 설치 점검
wandb login
```

## 할 일

`agent.py` 와 `config.yaml` 만 고친다.

- `obs_spec` / `action_spec` — 관측(`state` / `image`)과 액션 형태
- `setup` — 모델 (DQN / Dueling DQN / PPO)
- `preprocess` — 관측 → 신경망 입력
- `policy` — 행동 선택
- `reward` — 보상
- `train` — 학습 루프 (버퍼, 업데이트, wandb 로그)

`__init__` · `act` · `save` · `load` 는 고칠 수 없다. `load` 는 저장된 설정으로 `setup` 을 다시 부르므로 모델 구조는 `self.cfg` 만으로 정해져야 한다.
`config.yaml` 에는 상대 봇 목록과 하이퍼파라미터를 적는다.

```bash
python agent.py                   # 학습 → my_agent.pt
python agent.py --config dqn.yaml # 다른 설정 파일로
cell-arena-check agent.py         # 제출 전 점검
cell-arena-play                   # 직접 플레이
cell-arena-play --opponents gold agent.py   # 상대 지정 (봇 이름 또는 에이전트 파일)
```

제출: `agent.py` 를 `<이름>.py` 로 바꿔 가중치 `<이름>.pt` 와 함께 낸다. 다른 .py 를 import 하지 않는다.

## 게임 규칙

맵은 100 × 100 토러스. `s` = 세포 크기.

### 객체

| 객체 | 설명 |
|---|---|
| 세포 | 플레이어. 크기 100 에서 시작, 직경 `2·√(s/100)` |
| 밥 | 맵에 300개. 먹으면 밥 크기의 절반만큼 커진다. 먹히면 다른 곳에 다시 생긴다 |
| 세포밥 | 세포가 돌진할 때 흘리는 밥. 먹으면 크기의 절반만큼 커진다. 다시 생기지 않는다 |
| 블랙홀 | 15개. 크기 500 초과인 세포가 닿으면 −250. 발동하면 다른 곳에 다시 생긴다 |
| 화이트홀 | 15개. 크기 500 미만인 세포가 닿으면 +100. 발동하면 다른 곳에 다시 생긴다 |

### 조작

| 입력 | 동작 |
|---|---|
| 이동 | 8방향 또는 정지 (`continuous` 액션은 임의 각도). 클수록 느리고 방향 전환이 둔하다 |
| 돌진 | 이동 중에만. 클수록 빠르지만 4스텝마다 크기의 `1%·√(s/100)` 를 세포밥으로 흘린다. 크기 100 이하면 안 된다 |
| 직접 플레이 | `W` `A` `S` `D` 이동, `SPACE` 돌진, `TAB` 패널 세포 변경, `R` 리셋, `ESC` 종료 |

### 규칙

| 규칙 | 내용 |
|---|---|
| 포식 | 세포끼리 경계면이 닿으면 큰 쪽이 작은 쪽을 통째로 먹는다 (같은 크기면 무작위) |
| 사망 | 먹히거나, 블랙홀을 한 번에 여러 개 밟아 크기가 0 이하가 될 때 |
| 시야 | 한 변 `2·min(8·√(s/100), 25)` 인 정사각형. 그 밖은 보이지 않는다 |
| 승리 | 크기 4000 |
| 학습 중 | 죽으면 크기 100 으로 바로 리스폰하고 에피소드는 계속된다 |
| 대결 | 먹히면 탈락. 누군가 4000 에 도달하거나 한 명만 남으면 끝 |

## 관측

첫 차원은 배치 B (학습 = `num_envs`, 대결 = 1). 좌표는 내 세포 기준, y 는 아래가 +.

| 필드 | 형태 | 내용 |
|---|---|---|
| `self_state` | `(B, 5)` | `[크기, x, y, v_x, v_y]` |
| `objects` | `(B, M, 7)` | state 모드. 시야 안 객체 가까운 순 `[dx, dy, 크기, is_food, is_black_hole, is_white_hole, is_cell]` |
| `mask` | `(B, M)` | state 모드. 실제 객체면 True, 나머지는 0 패딩 |
| `image` | `(B, 5, R, R)` | image 모드. 채널 `[밥, 블랙홀, 화이트홀, 다른 세포, 나]`, 값 0/1 |

- `ObsSpec(mode="state", max_objects=32)` 또는 `ObsSpec(mode="image", resolution=64)`
- 이미지는 크기와 상관없이 R×R 이라 클수록 넓고 거칠게 본다
- 세포밥은 밥과 같은 `food` 로 보인다. 다른 세포의 속도는 주지 않는다

## 액션

| 모드 | 형태 | 내용 |
|---|---|---|
| `discrete` | `(B,)` ∈ [0, 18) | 0~8 = 정지·상·하·좌·우·좌상·우상·좌하·우하, 9~17 = 같은 순서 + 돌진 |
| `multibinary` | `(B, 5)` | `[상, 하, 좌, 우, 돌진]` |
| `continuous` | `(B, 3)` | `[θ, move, dash]`. θ 는 오른쪽 0, 아래 +π/2. move·dash 는 0.5 초과면 켜짐 |

## 사건 (`events`)

환경은 보상을 주지 않고 내 세포에게 일어난 사건만 준다. 각 `(B,)`.

| 필드 | 내용 |
|---|---|
| `size_before`, `size_after` | 스텝 전후 크기 |
| `food_mass` | 밥·세포밥으로 얻은 질량 |
| `white_hole`, `black_hole` | 발동 횟수 |
| `dash_cost` | 돌진으로 흘린 질량 |
| `kills`, `kill_mass` | 잡아먹은 세포 수와 얻은 질량 |
| `died`, `won` | 죽었는지, 4000 에 도달했는지 |
| `t` | 에피소드 경과 스텝 |

## 학습 env

`env.step(action)` 은 `obs, final_obs, events, terminated, truncated` 를 돌려준다.

- `terminated`: 누군가 4000 도달. 부트스트랩하지 않는다
- `truncated`: `max_steps` 도달. `final_obs` 로 부트스트랩한다
- 끝난 env 는 자동 리셋된다. `obs` 는 새 에피소드의 첫 관측, `final_obs` 는 리셋 전 마지막 관측
- 죽어도 에피소드는 끝나지 않는다 (`events.died`)
