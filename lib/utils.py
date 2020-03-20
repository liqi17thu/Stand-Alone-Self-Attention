import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR

import os
import shutil
import copy
import time

import logging
import logging.handlers

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def adjust_learning_rate(optimizer, epoch, original_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = original_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def save_checkpoint(state, is_best, save_path, filename):
    file_path = os.path.join(save_path, filename)
    torch.save(state, file_path)
    best_file_path = os.path.join(save_path, 'best_' + filename)
    if is_best:
        shutil.copyfile(file_path, best_file_path)


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


def make_divisible(v, divisor=8, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def int2list(val, repeat_time=1):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]


def get_logger(file_path, time=True):
    """ Make python logger """
    logger = logging.getLogger("train")
    if time:
        log_format = '%(asctime)s | %(message)s'
    else:
        log_format = '%(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

def get_attention_logger(file_path):
    """ Make python logger """
    logger = logging.getLogger("attention")
    log_format = '%(message)s'
    formatter = logging.Formatter(log_format)
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    return logger


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * \
            targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr / self.multiplier * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is not None and epoch > self.total_epoch:
            self.after_scheduler.step(epoch - self.total_epoch)
        else:
            super(GradualWarmupScheduler, self).step(epoch)


def get_scheduler(optimizer, n_iter_per_epoch, cfg):
    cosine_scheduler = CosineAnnealingLR(
        optimizer=optimizer, eta_min=0.000001,
        T_max=(cfg.train.epoch - cfg.train.start_epoch - cfg.optim.warmup_epoch) * n_iter_per_epoch)
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=cfg.optim.warmup_multiplier,
        total_epoch=cfg.optim.warmup_epoch * n_iter_per_epoch,
        after_scheduler=cosine_scheduler)
    return scheduler


def check_dir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder


multiply_adds = 1


def count_convNd(m, _, y):
    cin = m.in_channels

    kernel_ops = m.weight.size()[2] * m.weight.size()[3]
    ops_per_element = kernel_ops
    output_elements = y.nelement()

    # cout x oW x oH
    total_ops = cin * output_elements * ops_per_element // m.groups
    m.total_ops = torch.Tensor([int(total_ops)])


def count_linear(m, _, __):
    total_ops = m.in_features * m.out_features

    m.total_ops = torch.Tensor([int(total_ops)])


register_hooks = {
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    ######################################
    nn.Linear: count_linear,
    ######################################
    nn.Dropout: None,
    nn.Dropout2d: None,
    nn.Dropout3d: None,
    nn.BatchNorm2d: None,
    nn.AvgPool2d: None,
}

def profile(model, input_size, custom_ops=None):
    handler_collection = []
    custom_ops = {} if custom_ops is None else custom_ops

    def add_hooks(m_):
        if len(list(m_.children())) > 0:
            return

        m_.register_buffer('total_ops', torch.zeros(1))
        m_.register_buffer('total_params', torch.zeros(1))

        for p in m_.parameters():
            m_.total_params += torch.Tensor([p.numel()])

        m_type = type(m_)
        fn = None

        if m_type in custom_ops:
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
        else:
            # print("Not implemented for ", m_)
            pass

        if fn is not None:
            # print("Register FLOP counter for module %s" % str(m_))
            _handler = m_.register_forward_hook(fn)
            handler_collection.append(_handler)

    original_device = model.parameters().__next__().device
    training = model.training

    model.eval()
    model.apply(add_hooks)

    x = torch.zeros(input_size).to(original_device)
    with torch.no_grad():
        model(x)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        total_ops += m.total_ops
        total_params += m.total_params

    total_ops = total_ops.item()
    total_params = total_params.item()

    model.train(training).to(original_device)
    for handler in handler_collection:
        handler.remove()

    return total_ops, total_params


def count_net_flops(net, data_shape=(1, 3, 224, 224)):
    if isinstance(net, nn.DataParallel):
        net = net.module

    # net = copy.deepcopy(net)

    flop, _ = profile(net, data_shape)
    return flop


def get_net_device(net):
    return net.parameters().__next__().device


def rm_bn_from_net(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.forward = lambda x: x


def measure_net_latency(net, l_type='gpu8', fast=True, input_shape=(3, 224, 224), clean=False):
    if isinstance(net, nn.DataParallel):
        net = net.module

    # remove bn from graph
    rm_bn_from_net(net)

    # return `ms`
    if 'gpu' in l_type:
        l_type, batch_size = l_type[:3], int(l_type[3:])
    else:
        batch_size = 1

    data_shape = [batch_size] + list(input_shape)
    if l_type == 'cpu':
        if fast:
            n_warmup = 5
            n_sample = 10
        else:
            n_warmup = 50
            n_sample = 50
        if get_net_device(net) != torch.device('cpu'):
            net = copy.deepcopy(net).cpu()
    elif l_type == 'gpu':
        if fast:
            n_warmup = 5
            n_sample = 10
        else:
            n_warmup = 50
            n_sample = 50
    else:
        raise NotImplementedError
    images = torch.zeros(data_shape, device=get_net_device(net))

    measured_latency = {'warmup': [], 'sample': []}
    net.eval()
    with torch.no_grad():
        for i in range(n_warmup):
            inner_start_time = time.time()
            net(images)
            used_time = (time.time() - inner_start_time) * 1e3  # ms
            measured_latency['warmup'].append(used_time)
            if not clean:
                print('Warmup %d: %.3f' % (i, used_time))
        outer_start_time = time.time()
        for i in range(n_sample):
            net(images)
        total_time = (time.time() - outer_start_time) * 1e3  # ms
        measured_latency['sample'].append((total_time, n_sample))
    return total_time / n_sample, measured_latency


def count_parameters(net):
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_params


def get_net_info(net, input_shape=(3, 224, 224), measure_latency=None, logger=None, local_rank=0):
    net_info = {}
    if isinstance(net, nn.DataParallel):
        net = net.module

    # parameters
    net_info['params'] = count_parameters(net)

    # flops
    net_info['flops'] = count_net_flops(net, [1] + list(input_shape))

    # latencies
    latency_types = [] if measure_latency is None else measure_latency.split('#')
    for l_type in latency_types:
        latency, measured_latency = measure_net_latency(net, l_type, fast=False, input_shape=input_shape)
        net_info['%s latency' % l_type] = {
            'val': latency,
            'hist': measured_latency
        }

    if local_rank == 0:
        logger.info('Total training params: %.2fM' % (net_info['params'] / 1e6))
        logger.info('Total FLOPs: %.2fM' % (net_info['flops'] / 1e6))
        for l_type in latency_types:
            logger.info('Estimated %s latency: %.3fms' % (l_type, net_info['%s latency' % l_type]['val']))
