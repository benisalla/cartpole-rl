# Legacy notebooks

The original coursework notebooks this repository grew out of, copied verbatim (outputs
included). They use the old `gym==0.26.2` API and are kept for reference; the package in
`src/cartpole_rl` is the cleaned-up, tested version of the same ideas, plus the methods the
notebooks only listed as "to do" (Q-learning, DQN, actor-critic, A2C, PPO).

| notebook | what it does |
|---|---|
| [`hands_on_RL.ipynb`](hands_on_RL.ipynb) | CartPole helpers (frame capture, frame difference, live plots), then a vanilla policy gradient (REINFORCE) with three policies: a fully connected net on the 4-number state, a CNN on the rendered 400 x 600 frame, and a CNN with a residual block on augmented `img(t) - img(t-1)` difference images. Ends with a commented-out textbook REINFORCE. |
| [`Part_1_Policy_Gradient.ipynb`](Part_1_Policy_Gradient.ipynb) | The fully connected REINFORCE on the state alone, with unit checks of the model methods and a test function: reaches 500 / 500 on 50 test episodes. |
| [`Part_2_Policy_Gradient.ipynb`](Part_2_Policy_Gradient.ipynb) | The same fully connected REINFORCE training again (meant as the start of a second part; it installs stable-baselines3 and moviepy but does not use them). |
| [`CNN_Res_Diff_Between_Prev_Current.ipynb`](CNN_Res_Diff_Between_Prev_Current.ipynb) | The residual-CNN frame-difference policy from `hands_on_RL.ipynb`, on its own. |

`CNN_Res_Diff_Between_Prev_Current (1).ipynb`, the fifth file that came with these, is a
byte-for-byte duplicate of `CNN_Res_Diff_Between_Prev_Current.ipynb` (`diff` reports no
difference, both 500 814 bytes), so it is not copied here.

## What the notebooks got right

The fully connected REINFORCE with normalized reward-to-go (`calculate_discounted_rewards`,
`compute_policy_loss`) is correct and does solve CartPole (Part 1 tests at 500 / 500); it is
`--method reinforce` here. The frame-difference idea is the right instinct for pixels: a
single frame has no velocity in it. It is `--obs frame-diff` here.

## Bugs found, and how the package handles them

| # | where | bug | effect | fix in `cartpole_rl` |
|---|---|---|---|---|
| 1 | `hands_on_RL` CNN training + `evaluate_cnn_policy` | `get_img_obs(env, im_size=im_size)` but the function is `get_img_obs(env)` | `TypeError`: the single-frame CNN cell cannot run | one `PixelObservation` wrapper used by training and evaluation alike (`env.py`) |
| 2 | same | even with the call fixed, the raw `(3, 400, 600)` frame goes into a CNN whose `fc_dim = 32*7*7` was computed for 128 x 128 | shape error in `view(-1, fc_dim)` | frames are cropped and resized to 40 x 80; the CNN computes its flatten size from a dummy input (`CNNBody`) |
| 3 | frame-diff training | 64 augmented copies of one image per step, `rewards.extend([reward] * 64)`, then reward-to-go over the flat list | the 64 copies are treated as 64 consecutive time steps: the discount runs across copies (gamma^64 per real step) | one observation per step; returns computed per real time step |
| 4 | same | `log_probs` holds one `[64]` tensor per step while the returns list has `64 x T` entries; `zip` pairs step `t` with flat return `t` | step `t` is weighted by the return of an early step; most returns are never used | log-probs and returns are built from the same trajectory arrays, so they align by construction |
| 5 | same | the env executes the *mode* of 64 sampled actions, but all 64 log-probs are pushed up or down | actions that were never executed get the credit (or blame) | the executed action is the sampled action whose log-prob is used |
| 6 | all notebooks | epsilon-greedy mixed into a policy gradient: a random action's `log pi(a|s)` is used as if `a` came from pi | the gradient is no longer the on-policy gradient (no importance correction); it pushes up actions the policy did not choose | no epsilon in policy methods: the stochastic policy explores by itself; epsilon only in value methods, where it is correct |
| 7 | frame-diff training | augmentation with `RandomAffine(degrees=15)` and ImageNet `Normalize` | rotating a CartPole image by up to 15 degrees changes the pole angle, i.e. the very state the policy must read (the fail angle is 12 degrees) | no geometric augmentation; inputs are in [0, 1] (frame) and [-1, 1] (difference) |
| 8 | frame-diff | `get_image_diff` divides by the per-frame max of the difference | the scale of the input changes every step; a tiny motion looks as large as a big one | raw difference of two [0, 1] frames, plus the frame itself as a second channel so the position is still visible |
| 9 | frame-diff training | `normalize=False` with a summed loss | un-normalized returns (up to 1 / (1 - 0.98) = 50) summed over 64 x T terms: very large, unstable gradients | normalized advantages, mean (not sum) losses, gradient clipping where it matters |
| 10 | all | `F.softmax` in `forward`, then `torch.log(torch.clamp(prob, 1e-8))` | `log(softmax(x))` loses precision and the clamp hides zero probabilities | networks return logits; `Categorical(logits=...)` uses a stable log-softmax |
| 11 | all | dropout (0.2-0.5) active while sampling actions, off in evaluation | the policy that acts is not the one being evaluated; the log-probs are of a randomly thinned network | no dropout in the RL networks |
| 12 | `CNN_Res_Diff`, `hands_on_RL` env test cells | `for epi_idx in range(4)` never calls `env.reset()` between episodes | gym warns "calling step() after terminated"; episodes 2-4 are undefined behaviour | every episode starts with a seeded `reset` |
| 13 | all | "accuracy" = share of episodes reaching 500, evaluated on 5 episodes, on the training env | noisy, and stricter than the standard criterion | shared evaluation: greedy, fixed unseen seeds, 10 episodes periodically, 100 at the end, solved = mean >= 475 (`evaluation.md`) |
| 14 | `hands_on_RL` first cell | describes Tic-Tac-Toe and Lunar Lander | the text does not match the notebook (CartPole only) | README and docs describe what the code does |
| 15 | commented REINFORCE cell | every log-prob weighted by the whole episode return `R`, old `env.step` 4-tuple API | higher variance (an action is blamed for rewards before it) and would not run on gym 0.26 | reward-to-go, gymnasium 5-tuple API with `terminated` / `truncated` handled separately |
