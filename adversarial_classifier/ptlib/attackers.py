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
        self.hdf5filename_base = 'fgsm_'

    def _fgsm_attack(self, image, epsilon, data_grad):
        '''
        perturbed image = image + epsilon * sign(data_grad)
        '''
        data_grad_sign = data_grad.sign()
        perturbed_image = image + epsilon * data_grad_sign
        # TODO - do we need to re-normalize the image?
        return perturbed_image

    def _write_adversarial_output(
            self, labels, init_outputs, perturbed_outputs, adv_examples):
        labels = torch.stack(labels).numpy()
        init_outputs = torch.stack(init_outputs).numpy()
        perturbed_outputs = torch.stack(perturbed_outputs).numpy()
        adv_examples = torch.stack(adv_examples).numpy()
        print(labels.shape, init_outputs.shape, perturbed_outputs.shape,
              adv_examples.shape)

    def train_attack_for_single_epsilon(self, epsilon, short_test=False):
        LOGGER.info(
            'train_attack_for_single_epsilon for eps = {}'.format(epsilon))
        train_dl, _, _ = self.dm.get_data_loaders(batch_size=1)
        seen, correct = 0, 0
        true_labels = []
        initial_outputs = []
        perturbed_outputs = []
        adv_example_images = []
        self.model.eval()

        for iter_num, (inputs, labels) in enumerate(train_dl, 0):
            if short_test and iter_num >= 20:
                break
            seen += 1
            inputs.requires_grad = True
            output = self.model(inputs)
            initial_pred = output.max(1, keepdim=True)[1][0]
            LOGGER.debug('initial pred = {}, label = {}'.format(
                initial_pred.item(), labels.item()))
            # TODO - not sure about this, want to attack wrong labels also?
            if initial_pred.item() != labels.item():
                continue
            loss = self.criterion(output, labels)
            LOGGER.debug('loss = {}'.format(loss.item()))
            self.model.zero_grad()
            loss.backward()
            inputs_grad = inputs.grad.data
            perturbed_inputs = self._fgsm_attack(inputs, epsilon, inputs_grad)
            perturbed_output = self.model(perturbed_inputs)
            final_pred = perturbed_output.max(1, keepdim=True)[1][0]
            LOGGER.debug('final pred = {}, label = {}'.format(
                final_pred.item(), labels.item()))
            if final_pred.item() == labels.item():
                correct += 1
            true_labels.append(labels.squeeze().detach().cpu())
            initial_outputs.append(
                output.squeeze().detach().cpu())
            perturbed_outputs.append(
                perturbed_output.squeeze().detach().cpu())
            adv_example_images.append(
                perturbed_inputs.squeeze().detach().cpu())

        self._write_adversarial_output(
            true_labels, initial_outputs, perturbed_outputs,
            adv_example_images)
        final_accuracy = correct / float(seen)
        LOGGER.info('epsilon: {}, accuracy = {}/{} = {}'.format(
            epsilon, correct, seen, final_accuracy))
