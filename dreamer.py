import collections
import random
import torch
import torchvision.utils as vutils
from torch import optim, nn
from torch.utils.data import TensorDataset
from tqdm import tqdm
from util import create_folder, lr_cosine_policy, get_image_prior_losses, clip, DreamDataset


class DreamerFeatureHook(object):
    """
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    """
    def __init__(self, module):
        self.r_feature = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, inp, _):
        # hook function compute DeepInversion's feature distribution regularization
        nch = inp[0].shape[1]
        mean = inp[0].mean([0, 2, 3])
        var = inp[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


class DeepInversionDreamer(object):
    def __init__(self,
                 opt,
                 net=None,
                 criterion=None,
                 data_transform=None,
                 writer=None,
                 path='./final_images',
                 jitter=(30, 15),
                 network_output_function=lambda x: x):
        """
        :param opt: parameter options
        :param net: Pytorch model to be inverted
        :param criterion: cross-entropy criterion
        :param data_transform: data transform
        :param writer: summary writer
        :param path: path where to write temporal images and data
        :param jitter: amount of random shift applied to image at every iteration
        :param network_output_function:  function to be applied to the output of the network to get the output
        """

        # for reproducibility
        torch.manual_seed(torch.cuda.current_device())
        self.opt = opt
        self.net = net
        self.writer = writer
        self.data_transform = data_transform
        self.dream_person = opt.dream_person
        self.network_output_function = network_output_function
        self.image_resolution = [256, 128]
        self.do_flip = True

        self.ms = opt.ms  # memory size
        self.jitter = jitter
        self.criterion = criterion

        self.bn_reg_scale = opt.r_feature
        self.first_bn_multiplier = opt.first_bn_multiplier
        self.var_scale_l1 = opt.tv_l1
        self.var_scale_l2 = opt.tv_l2
        self.l2_scale = opt.l2
        self.lr = opt.dr_lr
        self.main_loss_multiplier = opt.main_loss_multiplier

        self.num_generations = 1
        self.final_data_path = path

        # Create folders for images and logs
        create_folder(self.final_data_path)

        # Create hooks for feature statistics
        self.loss_r_feature_layers = []
        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.loss_r_feature_layers.append(DreamerFeatureHook(module))

    def dream_images(self):
        net = self.net
        criterion = self.criterion
        img_original = self.image_resolution
        logit_dim = net.classifier.output_dim
        if logit_dim > self.ms:
            self.ms = logit_dim
        if self.dream_person*logit_dim > self.ms:
            gen_num = (self.ms // logit_dim) * logit_dim
        else:
            gen_num = self.dream_person * logit_dim

        assert gen_num > 0, 'The generation number of images should be larger than zero.'
        self.dream_person = gen_num // logit_dim
        assert self.dream_person > 0, 'The generation number of images for each identity should be larger than zero.'
        num_batch, rest = gen_num // (self.dream_person * 64), gen_num % (self.dream_person * 64)
        if rest == 1:
            split_list = [self.dream_person * 64] * num_batch
            split_list[-1] += 1
        else:
            split_list = [self.dream_person * 64] * num_batch + [rest]
        assert sum(split_list) == gen_num
        inputs = torch.split(torch.randn((self.dream_person*logit_dim, 3, img_original[0], img_original[1]),
                                         dtype=torch.float), split_list)
        targets = []
        for i in range(logit_dim):
            targets += [i] * self.dream_person
        # random.shuffle(targets)
        targets = torch.split(torch.LongTensor(targets), split_list)

        pooling_function = nn.modules.pooling.AvgPool2d(kernel_size=2)
        optimizer = None
        iterations_per_layer = self.opt.iteration
        best_inputs = []
        best_inputs_ = None

        for step, (ins, tas) in tqdm(enumerate(zip(inputs, targets)), desc='2. Getting dream images'):
            ins = ins.cuda().requires_grad_()
            tas = tas.cuda()
            bsz = tas.shape[0]
            best_cost = 1e5
            iteration = 0
            for lr_it, lower_res in enumerate([2, 1]):
                optimizer = optim.Adam([ins], lr=self.lr, betas=(0.5, 0.9), eps=1e-8)
                lim_0, lim_1 = self.jitter[0] // lower_res, self.jitter[1] // lower_res
                lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)
                for iteration_loc in range(iterations_per_layer):
                    iteration += 1
                    # learning rate scheduling
                    lr_scheduler(optimizer, iteration_loc, iteration_loc)

                    # perform down sampling if needed
                    if lower_res != 1:
                        ins_jit = pooling_function(ins)
                    else:
                        ins_jit = ins

                    # apply random jitter offsets
                    off1 = random.randint(-lim_0, lim_0)
                    off2 = random.randint(-lim_1, lim_1)
                    ins_jit = torch.roll(ins_jit, shifts=(off1, off2), dims=(2, 3))

                    # flipping
                    flip = random.random() > 0.5
                    if flip and self.do_flip:
                        ins_jit = torch.flip(ins_jit, dims=(3,))

                    # forward pass
                    optimizer.zero_grad()
                    net.zero_grad()
                    _, outputs, _ = net(ins_jit)
                    outputs = self.network_output_function(outputs)

                    # R_cross classification loss
                    loss = criterion(outputs, tas)

                    # R_prior losses
                    loss_var_l1, loss_var_l2 = get_image_prior_losses(ins_jit)

                    # R_feature loss
                    rescale = [self.first_bn_multiplier] + [1. for _ in range(len(self.loss_r_feature_layers)-1)]
                    loss_r_feature = sum(
                        [mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.loss_r_feature_layers)]
                    )

                    # l2 loss on images
                    loss_l2 = torch.norm(ins_jit.view(bsz, -1), dim=1).mean()

                    # combining losses
                    loss_aux = \
                        self.var_scale_l2 * loss_var_l2 + \
                        self.var_scale_l1 * loss_var_l1 + \
                        self.bn_reg_scale * loss_r_feature + \
                        self.l2_scale * loss_l2

                    loss = self.main_loss_multiplier * loss + loss_aux

                    # do image update
                    loss.backward()
                    self.writer.add_scalar("total loss", loss.item(), global_step=iteration)
                    self.writer.add_scalar("loss_r_feature loss", loss_r_feature.item(), global_step=iteration)
                    self.writer.add_scalar("main criterion loss", criterion(outputs, tas).item(), global_step=iteration)
                    optimizer.step()

                    # clip color out layers
                    ins.data = clip(ins.data)

                    if best_cost > loss.item() or iteration == 1:
                        best_inputs_ = ins.data.clone().detach()
                        best_cost = loss.item()
            best_inputs.append(best_inputs_)

        best_inputs = torch.cat(best_inputs, dim=0)
        targets = torch.cat(targets, dim=0)
        vutils.save_image(
            best_inputs, '{}/output_{:05d}.png'.format(self.final_data_path, gen_num),
            normalize=True, scale_each=True, nrow=self.dream_person*int(64)
        )
        self.writer.add_images('Dreamer images', best_inputs[::self.dream_person*10], self.num_generations)
        self.num_generations += 1

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)
        dream_dataset = DreamDataset(best_inputs.detach().cpu(), targets.detach().cpu(),
                                     transform=self.data_transform)
        dream_dataloader = torch.utils.data.DataLoader(dream_dataset, batch_size=self.opt.batch_size,
                                                       shuffle=True, num_workers=self.opt.num_workers, pin_memory=True)
        return dream_dataloader
