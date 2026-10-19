import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)


@pytest.fixture
def tmp_out(tmp_path):
    return str(tmp_path / "run")
