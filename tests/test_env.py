import numpy as np
import pytest

from cartpole_rl.env import Discretizer, VecEnv, make_env, preprocess_frame


def test_state_env_shapes_and_seeding():
    a, b = make_env("state"), make_env("state")
    oa, _ = a.reset(seed=3)
    ob, _ = b.reset(seed=3)
    assert oa.shape == (4,)
    np.testing.assert_array_equal(oa, ob)


@pytest.mark.parametrize("mode,channels", [("frame", 1), ("frame-diff", 2)])
def test_pixel_modes(mode, channels):
    env = make_env(mode, frame_height=40, frame_width=80)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (channels, 40, 80) and obs.dtype == np.float32
    assert obs.min() >= -1.0 and obs.max() <= 1.0
    if mode == "frame-diff":
        assert np.abs(obs[1]).max() == 0  # no previous frame yet: zero difference
        for _ in range(3):
            obs, *_ = env.step(1)
        assert np.abs(obs[1]).max() > 0  # the cart moved: the difference channel lights up
    env.close()


def test_preprocess_objects_are_bright():
    env = make_env("state", render=True)
    env.reset(seed=0)
    f = preprocess_frame(env.render())
    assert f.shape == (40, 80)
    assert f.max() > 0.5 and np.median(f) < 0.05  # dark background, bright cart/pole


def test_discretizer_range_and_monotone():
    d = Discretizer((1, 6, 12, 12))
    assert d.n_states == 864
    lo, hi = d(np.array([-9, -9, -9, -9])), d(np.array([9, 9, 9, 9]))
    assert lo == 0 and hi == d.n_states - 1
    seen = {d(np.random.default_rng(i).uniform(-3, 3, 4)) for i in range(200)}
    assert all(0 <= s < d.n_states for s in seen)


def test_vec_env_autoreset_reports_episodes():
    venv = VecEnv(3, seed=0)
    obs = venv.reset()
    assert obs.shape == (3, 4)
    finished = []
    for _ in range(200):
        obs, r, term, trunc, final_obs, done = venv.step(np.ones(3, dtype=int))  # always push right: falls fast
        finished += done
        assert r.shape == (3,) and final_obs.shape == (3, 4)
    assert len(finished) >= 3 and all(ret == length for ret, length in finished)
