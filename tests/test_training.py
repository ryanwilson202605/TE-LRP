"""Regression tests for training and checkpoint restoration."""
import torch

from te_lrp.training.trainer import Trainer


def test_train_save_resume(config, batch):
    """Verify model, optimizer, and scheduler restoration after training.

    Inputs: config (DictConfig), batch (dict): test fixtures.
    Returns: None."""
    trainer = Trainer(config=config)
    before = next(trainer.model.parameters()).detach().clone()
    trainer.fit(train_loader=[batch], val_loader=[batch], max_epochs=1)
    assert not torch.equal(before, next(trainer.model.parameters()))
    restored = Trainer(config=config)
    restored.load_checkpoint(checkpoint_path=config.latest_ckpt)
    assert restored.current_epoch == 1
    assert restored.global_step == 1
    assert restored.scheduler.state_dict() == trainer.scheduler.state_dict()
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[key])


def test_likelihood_weights_favor_fitting_mode(config):
    """Check that posterior weights favor the higher-likelihood mode.

    Inputs: config (DictConfig): small-model fixture.
    Returns: None."""
    from te_lrp.models.autobot import Criterion
    config.entropy_weight = 0
    config.kl_weight = 0
    config.use_FDEADE_aux_loss = False
    criterion = Criterion(config=config)
    trajectories = torch.zeros(1, 2, 60, 5)
    trajectories[..., 2:4] = 1
    trajectories[:, 1, :, :2] = 10
    prediction = {'predicted_trajectory': trajectories, 'predicted_probability': torch.full((1, 2), 0.5)}
    gt = torch.zeros(1, 60, 3)
    gt[..., -1] = 1
    loss = criterion(out=prediction, gt=gt, center_gt_final_valid_idx=torch.tensor([59]))
    torch.testing.assert_close(loss, torch.tensor(120.) * torch.log(torch.tensor(2.)))
