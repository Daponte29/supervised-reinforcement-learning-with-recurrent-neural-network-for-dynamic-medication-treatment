import numpy as np

from src.data import (
    align_and_filter_data,
    align_lab_to_action,
    build_loaders,
    infer_feature_dims,
    my_collate_fn,
    truncate_to_window,
)
from tests.conftest import DEMO_SIZE, DI_SIZE, LAB_SIZE, MED_SIZE, TIME_STAMP


def test_align_and_filter_intersects_and_rewards():
    action = {1: np.ones((3, MED_SIZE)), 2: np.ones((3, MED_SIZE)), 3: np.ones((3, MED_SIZE))}
    lab = {1: np.ones((3, LAB_SIZE)), 2: np.ones((3, LAB_SIZE))}  # missing key 3
    demo = {1: np.ones(DEMO_SIZE), 2: np.ones(DEMO_SIZE), 3: np.ones(DEMO_SIZE)}
    diag = {1: np.zeros(DI_SIZE), 2: np.zeros(DI_SIZE), 3: np.zeros(DI_SIZE)}
    label = {1: 0, 2: 1, 3: 0}

    a, _, _, _, _, reward = align_and_filter_data(action, lab, demo, label, diag)
    assert set(a.keys()) == {1, 2}  # key 3 dropped (absent from lab)
    assert reward[1][0] == 10.0  # survived
    assert reward[2][0] == -10.0  # died


def test_align_lab_to_action_pads_and_truncates():
    action = {1: np.ones((8, MED_SIZE)), 2: np.ones((8, MED_SIZE))}
    lab = {1: np.ones((10, LAB_SIZE)), 2: np.ones((5, LAB_SIZE))}
    fixed = align_lab_to_action(action, lab)
    assert fixed[1].shape == (8, LAB_SIZE)  # truncated from 10
    assert fixed[2].shape == (8, LAB_SIZE)  # front-padded from 5
    assert np.all(fixed[2][:3] == 0)  # padding at the front


def test_truncate_to_window_filters_short_sequences():
    action = {1: np.ones((10, MED_SIZE)), 2: np.ones((6, MED_SIZE))}
    lab = {1: np.ones((10, LAB_SIZE)), 2: np.ones((6, LAB_SIZE))}
    demo = {1: np.ones(DEMO_SIZE), 2: np.ones(DEMO_SIZE)}
    diag = {1: np.zeros(DI_SIZE), 2: np.zeros(DI_SIZE)}
    label = {1: 0, 2: 0}
    reward = {1: [10.0] * 10, 2: [10.0] * 6}

    a, lab_t, _, _, _, _ = truncate_to_window(action, lab, demo, label, diag, reward, TIME_STAMP)
    assert set(a.keys()) == {1}  # only the length-10 admission survives (> 8)
    assert a[1].shape[0] == TIME_STAMP
    assert lab_t[1].shape[0] == TIME_STAMP


def test_dataset_getitem_shapes(dataset):
    states, actions, rewards, dones, diseases, demos = dataset[0]
    assert np.asarray(states).shape == (TIME_STAMP, LAB_SIZE)
    assert np.asarray(actions).shape == (TIME_STAMP, MED_SIZE)
    assert np.asarray(diseases).shape == (DI_SIZE,)
    assert np.asarray(demos).shape == (DEMO_SIZE,)
    assert len(dones) == TIME_STAMP and dones[-1] == 1


def test_infer_feature_dims(dataset):
    dims = infer_feature_dims(dataset)
    assert dims == {"lab_size": LAB_SIZE, "med_size": MED_SIZE, "demo_size": DEMO_SIZE, "di_size": DI_SIZE}


def test_collate_batch_shapes(dataset):
    batch = [dataset[i] for i in range(4)]
    states, actions, rewards, next_states, dones, diseases, demos = my_collate_fn(batch)
    assert states.shape == (4, TIME_STAMP, LAB_SIZE)
    assert actions.shape == (4, TIME_STAMP, MED_SIZE)
    assert next_states.shape == (4, TIME_STAMP, LAB_SIZE)
    assert rewards.shape == (4, TIME_STAMP)
    assert dones.shape == (4, TIME_STAMP)
    assert diseases.shape == (4, DI_SIZE)
    assert demos.shape == (4, DEMO_SIZE)
    # next_state terminal timestep is zeroed
    assert next_states[:, -1, :].abs().sum().item() == 0.0


def test_build_loaders_splits(dataset, data_cfg):
    train_loader, val_loader, test_loader = build_loaders(dataset, data_cfg)
    total = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
    assert total == len(dataset)
    assert len(train_loader.dataset) > 0
