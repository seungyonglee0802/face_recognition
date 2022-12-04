import torch

def Scheduler(optimizer, test_interval, max_epoch, lr_decay, **kwargs):

	sche_fn = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.00001)

	lr_step = 'epoch'

	print('Initialised cosineWarmRestart LR scheduler')

	return sche_fn, lr_step