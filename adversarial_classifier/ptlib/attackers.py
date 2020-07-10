'''
FGSM attack code strongly inspired by PyTorch docs example:
    https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
'''
import logging
import torch
from ptlib.model_handlers import ModelHandlerBase

LOGGER = logging.getLogger(__name__)


class FGSMAttacker(ModelHandlerBase):
    def __init__(self, data_manager, model, ckpt_path, epsilons, log_freq=100):
        '''
        epsilons - scalar or list (iterable) of scales to apply to attack
        '''
        super(FGSMAttacker, self).__init__(
            data_manager, model, ckpt_path, log_freq)
        try:
            self.epsilons = list(epsilons)
        except TypeError as e:
            # assume in this case, epsilons is not iterable
            LOGGER.warning(e)
            self.epsilons = [epsilons]

    def fgsm_attack(image, epsilon, data_grad):
        '''
        perturbed image = image + epsilon * sign(data_grad)
        '''
        data_grad_sign = data_grad.sign()
        perturbed_image = image + epsilon * data_grad_sign
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image

    def attack_for_single_epsilon(self, epsilon):
        pass
