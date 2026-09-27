# Cell Arena 학생 에이전트 가이드

`agent.py` 한 파일이 **학습 코드이자 제출물**이다. `config.yaml` 에 상대 구성과 하이퍼파라미터를 적는다.
고칠 수 있는 파일은 이 두 개뿐이고, `agent.py` 안에서도 `StudentAgent` 가 정한 메소드는 바꿀 수 없다.

```bash
pip install -r requirements.txt
cell-arena-doctor                  # 설치·하드웨어 점검 (처음 한 번)
python agent.py                    # 학습 → my_agent.pt
python agent.py --config dqn.yaml  # 다른 설정 파일로
cell-arena-check agent.py          # 제출 전 점검
cell-arena-play                    # 게임 직접 플레이 (W/A/S/D 이동, SPACE 돌진, TAB 패널 세포 변경, R 리셋, ESC 종료)
cell-arena-play --opponents gold agent.py   # 상대 지정: 봇 이름 또는 내 에이전트 파일 (가중치 필요)
```

제출: `agent.py` 를 본인 이름으로 바꿔 가중치 파일과 함께 낸다 (또는 이 레포를 그대로 제출). 파일 하나에 모든 코드를 담는다 (다른 .py 를 import 하지 않는다).

## 무엇을 채우나

| | 메소드 | 할 일 |
|---|---|---|
| ① | `obs_spec`, `action_spec` | 관측 형태 (`state` / `image`), 액션 형태 (`discrete` / `multibinary` / `continuous`) |
| ② | `setup()` | 모델 — 인코더와 알고리즘별 머리 (DQN: Q / Dueling: V + A / PPO: 정책 + 가치). `self.cfg`, `self.device` 사용 가능 |
| ③ | `preprocess(obs)` | 관측 → 신경망 입력 (정규화, 피처, 프레임 쌓기 ...) |
| ④ | `policy(x, explore)` | 신경망 입력 → 행동. `explore=False` 가 대결에서 쓰는 행동 |
| ⑤ | `reward(events, obs)` | 보상 설계 |
| | `reset(done)` (선택) | 프레임 스택·RNN 기억 초기화 |
| ⑥ | `train(cfg)` | 학습 루프 — 버퍼, 업데이트 주기, 부트스트랩, 로그(wandb) 전부 |

**바꿀 수 없는 것** (재정의하면 파일을 불러올 때 오류):

| 메소드 | 하는 일 |
|---|---|
| `__init__(cfg, device=None)` | `self.cfg`, `self.device` (cuda > mps > cpu) 를 준비하고 `setup()` 호출 |
| `act(obs, explore=False)` | `preprocess` → `policy` 를 그래디언트 없이 실행하고 액션 형태를 검사. 대결장은 `explore=False`, B=1 로 부른다 |
| `save(path=None)` | 속성인 `torch.nn.Module` 전부 + 설정 + 입출력 형태를 저장. 기본 경로는 이 파일 폴더의 `weights` |
| `load(path)` | 저장된 설정으로 `setup()` 을 다시 부른 뒤 가중치를 채우고 eval 모드로 (CPU) |

- `load` 는 저장 당시 설정으로 `setup()` 을 다시 부른다 → **모델 구조는 `self.cfg` 만으로 정해져야 한다.**
- 대결장은 파일을 import 해서 클래스만 쓴다. 학습 코드는 `if __name__ == "__main__":` 아래에 둔다.

## config.yaml

```yaml
opponents: [bronze, bronze, bronze, silver, silver, gold, diamond]  # 봇 이름 또는 에이전트 파일 (.py)
num_envs: 64        # 배치 B
max_steps: 1000     # 에피소드 길이 상한 T (null = 무한)
seed: 0
total_samples: 5_000_000   # ↓ 여기부터는 자유. 코드에서 cfg.total_samples
lr: 3.0e-4
```

위 네 키는 `make_env` 가 읽는다 (필수). 나머지는 마음대로 추가하고 `cfg.<키>` 로 읽는다 (`cfg.get("키", 기본값)` 도 된다).
게임 규칙은 바꿀 수 없다. 봇 이름(`bronze`/`silver`/`gold`/`diamond`)은 `cell_arena` 패키지에 내장돼 있다.

## 관측 (입력)

모든 배열의 첫 차원은 배치 B (학습 = `num_envs`, 대결 = 1). 좌표는 내 세포 중심 기준, 단위 게임유닛, **y 는 아래가 +**.

| 필드 | 형태 | 내용 |
|---|---|---|
| `obs.self_state` | `(B, 5)` float32 | `[size, x, y, v_x, v_y]` — 크기, 맵 좌표 (0~100, 토러스), 속도. 두 모드 모두 준다 |
| `obs.objects` | `(B, M, 7)` float32 | state 모드. 시야 안 객체 가까운 순 `[dx, dy, 크기, is_food, is_black_hole, is_white_hole, is_cell]` |
| `obs.mask` | `(B, M)` bool | state 모드. 실제 객체 자리 True (나머지는 0 패딩) |
| `obs.image` | `(B, 5, R, R)` uint8 | image 모드. 채널 `[food, black_hole, white_hole, other_cell, self]`, 값 0/1, `[채널, y, x]` |

- 시야는 한 변 `2h` 인 정사각형, `h = min(8·√(size/100), 25)`. 이미지는 크기와 상관없이 R×R 이라 **클수록 넓고 거칠게** 본다.
- 객체의 `크기`는 원시 크기 그대로다 (직경이 아님). 직경이 필요하면 `d = 2·√(크기/100)`. 세포밥(돌진할 때 흘린 밥)은 관측에서 밥과 같은 `food` 로 준다.
- 이미지: 픽셀 중심이 원 안이면 1, 픽셀보다 작은 객체도 중심 픽셀은 1. 채널마다 가까운 순으로 밥 128개, 홀·세포 16개까지 그린다.
- **다른 세포의 속도는 주지 않는다** → 여러 프레임을 쌓거나 RNN. 전역 맵·시야 밖 정보도 없다. 봇도 똑같은 관측만 받는다.
- `ObsSpec(mode="state", max_objects=32)`, `ObsSpec(mode="image", resolution=64)`.

## 액션 (출력)

| 모드 | 형태 | 내용 |
|---|---|---|
| `discrete` | `(B,)` 정수 ∈ [0, 18) | 0~8 = 정지·상·하·좌·우·좌상·우상·좌하·우하, 9~17 = 같은 순서 + 돌진 |
| `multibinary` | `(B, 5)` bool | `[상, 하, 좌, 우, 돌진]` |
| `continuous` | `(B, 3)` float | `[θ, move, dash]` — θ = atan2(dy, dx) (오른쪽 0, 아래 +π/2), move·dash 는 0.5 초과면 켜짐 |

돌진: 크기 100 초과이고 움직일 때만. 클수록 빠르지만 (`1.1·(s/100)^0.2`) 4스텝마다 크기의 `1%·√(s/100)` 를 세포밥으로 흘리고 선회가 둔해진다.

## 사건 (보상 재료) — `reward(events, obs)`

환경은 보상을 주지 않고 **내 세포에게 일어난 일**만 준다. 각 `(B,)`.

| `events.` | 내용 |
|---|---|
| `size_before`, `size_after` | 스텝 전후 크기 (먹혀 죽으면 직전 크기, 블랙홀로 0 이하가 되어 죽으면 그 값) |
| `food_mass` | 밥·세포밥으로 얻은 질량 |
| `white_hole`, `black_hole` | 발동 횟수 (+100 / −250, 크기 500 미만 / 초과일 때만) |
| `dash_cost` | 돌진으로 흘린 질량 |
| `kills`, `kill_mass` | 잡아먹은 세포 수와 얻은 질량 |
| `died`, `won` | 죽었는지 (먹힘 또는 블랙홀), 승리 크기 4000 을 넘었는지 |
| `t` | 에피소드 경과 스텝 |

## 학습 env — `make_env(cfg, agent)`

`env.reset() -> Observation`, `env.step(action) -> StepOutput(obs, final_obs, events, terminated, truncated)`.

- 한 env = 내 세포 1 + `opponents`. 하드웨어는 자동 (JAX CPU/GPU, JAX 가 없으면 같은 결과의 NumPy env — 느리니 `cell-arena-doctor` 로 확인)
- **`terminated`**: 어떤 세포든 4000 도달 → 게임 끝. 부트스트랩하지 않는다
- **`truncated`**: `max_steps` 도달 → 잘렸을 뿐. `final_obs` 로 부트스트랩한다
- **죽음은 에피소드를 끝내지 않는다**: 나든 상대든 모든 시야 밖에서 크기 100 으로 즉시 리스폰 (`events.died`)
- 끝난 env 는 자동 리셋: `obs` 는 새 에피소드의 첫 관측, `final_obs` 는 리셋 전 관측

## 대결

최종 평가는 강사가 제출된 에이전트들끼리 토너먼트로 진행한다. 대결장은 `act(obs)` 를 매 스텝 B=1, `explore=False` 로 부른다 — 그 외에는 신경 쓸 것 없다.

## 로그 (wandb)

`train()`에 기본 로깅이 이미 동작하는 코드로 들어 있다 (`reward_step`, `episode_return`, `size`, `deaths`,
`samples_per_sec` — `log_every` 샘플마다 `run.log`). 프레임워크가 아니라 `agent.py` 안의 평범한 코드라 loss·Q값·
엔트로피 등 원하는 지표를 자유롭게 추가·수정한다. 처음 한 번 `wandb login`.
인터넷이 없으면 `WANDB_MODE=offline python agent.py` 로 학습하고 나중에 `wandb sync wandb/offline-run-*`.
