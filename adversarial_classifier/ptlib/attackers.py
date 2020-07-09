'''
FGSM attack code strongly inspired by PyTorch docs example:
    https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
'''
import torch
from ptlib.model_handlers import ModelHandlerBase


class FGSMAttacker(ModelHandlerBase):
    def __init__(self, model, ckpt_path, epsilons):
        super(FGSMAttacker, self).__init__(model, ckpt_path)
        self.epsilons = list(epsilons)

    def fgsm_attack(image, epsilon, data_grad):
        '''
        perturbed image = image + epsilon * sign(data_grad)
        '''
        data_grad_sign = data_grad.sign()
        perturbed_image = image + epsilon * data_grad_sign
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image
