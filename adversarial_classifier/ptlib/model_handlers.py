import logging
import torch

LOGGER = logging.getLogger(__name__)


class ModelHandlerBase(object):
    def __init__(self, model, ckpt_path):
        self.model = model
        self.ckpt_path = ckpt_path
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

    def _save_state(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.ckpt_path)

    def restore_model_and_optimizer(self):
        try:
            checkpoint = torch.load(self.ckpt_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            LOGGER.info('Loaded checkpoint from {}'.format(self.ckpt_path))
        except FileNotFoundError:
            LOGGER.info('No checkpoint found...')

        LOGGER.debug('Model state dict:')
        for param_tensor in self.model.state_dict():
            LOGGER.debug(str(param_tensor) + '\t'
                         + str(self.model.state_dict()[param_tensor].size()))
        LOGGER.debug('Optimizer state dict:')
        for var_name in self.optimizer.state_dict():
            LOGGER.debug(str(var_name) + '\t'
                         + str(self.optimizer.state_dict()[var_name]))

        self.model.to(self.device)
