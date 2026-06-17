from __future__ import annotations

import numpy as np
import pytest

from rsl_rl.algorithms.plugins.amp.motion_loader import AMPLoader


def _write_motion(path, body_names=None) -> None:
    kwargs = {}
    if body_names is not None:
        kwargs["body_names"] = np.asarray(body_names)
    np.savez(
        path,
        fps=np.asarray([50.0], dtype=np.float32),
        joint_pos=np.zeros((2, 1), dtype=np.float32),
        joint_vel=np.zeros((2, 1), dtype=np.float32),
        body_pos_w=np.asarray(
            [
                [[30.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                [[30.0, 0.0, 0.0], [11.0, 0.0, 0.0], [21.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        body_quat_w=np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (2, 3, 1)),
        body_lin_vel_w=np.zeros((2, 3, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((2, 3, 3), dtype=np.float32),
        **kwargs,
    )


def test_amp_loader_uses_npz_body_names(tmp_path) -> None:
    path = tmp_path / "motion.npz"
    _write_motion(path, body_names=["b3", "anchor", "b2"])

    loader = AMPLoader(
        str(path),
        body_names=["anchor", "b2"],
        anchor_name="anchor",
        all_body_names=["anchor", "b2", "b3"],
        device="cpu",
    )

    assert loader.observation_dim == 30
    assert np.allclose(loader._body_pos_b_list[0][0, :, 0].cpu().numpy(), [0.0, 10.0])


def test_amp_loader_allows_anchor_outside_selected_body_names_with_metadata(tmp_path) -> None:
    path = tmp_path / "motion.npz"
    _write_motion(path, body_names=["b3", "anchor", "b2"])

    loader = AMPLoader(
        str(path),
        body_names=["b3", "b2"],
        anchor_name="anchor",
        all_body_names=["anchor", "b2", "b3"],
        device="cpu",
    )

    assert loader.observation_dim == 30
    assert np.allclose(loader._body_pos_b_list[0][0, :, 0].cpu().numpy(), [20.0, 10.0])
    assert loader._anchor_indexes == 1


def test_amp_loader_legacy_selected_order(tmp_path) -> None:
    path = tmp_path / "motion.npz"
    np.savez(
        path,
        fps=np.asarray([50.0], dtype=np.float32),
        joint_pos=np.zeros((2, 1), dtype=np.float32),
        joint_vel=np.zeros((2, 1), dtype=np.float32),
        body_pos_w=np.asarray(
            [
                [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                [[11.0, 0.0, 0.0], [21.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        body_quat_w=np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (2, 2, 1)),
        body_lin_vel_w=np.zeros((2, 2, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((2, 2, 3), dtype=np.float32),
    )

    loader = AMPLoader(
        str(path),
        body_names=["anchor", "b2"],
        anchor_name="anchor",
        all_body_names=["anchor", "b2", "b3"],
        device="cpu",
    )

    assert np.allclose(loader._body_pos_b_list[0][0, :, 0].cpu().numpy(), [0.0, 10.0])


def test_amp_loader_rejects_ambiguous_legacy_body_dim(tmp_path) -> None:
    path = tmp_path / "motion.npz"
    np.savez(
        path,
        fps=np.asarray([50.0], dtype=np.float32),
        joint_pos=np.zeros((2, 1), dtype=np.float32),
        joint_vel=np.zeros((2, 1), dtype=np.float32),
        body_pos_w=np.zeros((2, 3, 3), dtype=np.float32),
        body_quat_w=np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (2, 3, 1)),
        body_lin_vel_w=np.zeros((2, 3, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((2, 3, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="attach_npz_names"):
        AMPLoader(
            str(path),
            body_names=["anchor", "b2"],
            anchor_name="anchor",
            all_body_names=["anchor", "b2", "b3", "b4"],
            device="cpu",
        )
