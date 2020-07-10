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

    def fgsm_attack(self, image, epsilon, data_grad):
        '''
        perturbed image = image + epsilon * sign(data_grad)
        '''
        data_grad_sign = data_grad.sign()
        perturbed_image = image + epsilon * data_grad_sign
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image

    # TODO - add open and close HDF5 file methods for logging adv images

    def train_attack_for_single_epsilon(self, epsilon, short_test=False):
        LOGGER.info(
            'train_attack_for_single_epsilon for eps = {}'.format(epsilon))
        train_dl, _, _ = self.dm.get_data_loaders(batch_size=1)
        correct = 0
        adv_examples = []

        for iter_num, (inputs, labels) in enumerate(train_dl, 0):
            if short_test and iter_num >= 20:
                break
            inputs.requires_grad = True
            output = self.model(inputs)
            initial_pred = output.max(1, keepdim=True)[1][0]
            LOGGER.debug('initial pred = {}, label = {}'.format(
                initial_pred.item(), labels.item()))
            # TODO - not sure about this, want to attack wrong labels also?
            # if initial_pred.item() != labels.item():
            #     continue
            loss = self.criterion(output, labels)
            LOGGER.debug('loss = {}'.format(loss.item()))
            self.model.zero_grad()
            loss.backward()
            inputs_grad = inputs.grad.data
            perturbed_inputs = self.fgsm_attack(inputs, epsilon, inputs_grad)
            LOGGER.debug('perturbed shape = {}'.format(perturbed_inputs.shape))
