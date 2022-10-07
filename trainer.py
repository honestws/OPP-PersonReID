import torch
import torch.nn.functional as F
from tqdm import tqdm

from builder import create_optimizer
from util import average, get_targets, interleave, get_assigned_label, linear_rampup


class Trainer(object):
    def __init__(self, opt, optimizer_con, optimizer_ema,
                 model, ema_model, writer, con_loss, mix_loss,
                 cross_entropy_loss, continual_index_list):
        self.opt = opt
        self.optimizer_con = optimizer_con
        self.optimizer_ema = optimizer_ema
        self.model = model
        self.ema_model = ema_model
        self.writer = writer
        self.con_loss = con_loss
        self.mix_loss = mix_loss
        self.cross_entropy_loss = cross_entropy_loss
        self.continual_index_list = continual_index_list
        self.feat_buffer = None
        self.len_continual_index_list = len(continual_index_list)
        self.feat_cat = None

    def create_feature_buffer(self, dream_dataloader):
        buffer = {}
        for images, labels in tqdm(dream_dataloader, desc='3. Creating feature buffer'):
            imgs = images[0].cuda()
            _, _, x = self.ema_model(imgs)
            for lab in labels:
                if lab.item() in buffer.keys():
                    continue
                else:
                    same_cls = (labels == lab).detach().cpu().numpy().tolist()
                    # mask = [(p & q) for p, q in zip(same_cls, curr_idx)]

                    n = sum(same_cls)
                    s = torch.sum(x[same_cls, :], dim=0, keepdim=True).clone().detach()
                    buffer[lab.item()] = []
                    buffer[lab.item()].append(s)
                    buffer[lab.item()].append(n)
        feat_buffer = []
        for k, v in average(buffer).items():
            feat_buffer.append(v)
        self.feat_buffer = torch.cat(feat_buffer, dim=0)
        assert self.feat_buffer.size(0) == self.ema_model.classifier.output_dim

    def train_within_camera_view(self, train_dataloader, epoch, ith, lab_dict, camera_person_list, optimizer_cro):
        for step, (images, labels, indices) in enumerate(train_dataloader):
            reassigned_labels = get_assigned_label(labels, lab_dict)
            ims = images[0].cuda()
            reassigned_labels = reassigned_labels.cuda()
            z, logit, _ = self.model(ims)

            curr_idx = [i in self.continual_index_list[ith] for i in indices]
            assert len(curr_idx) == sum(curr_idx)

            con_loss = self.con_loss(z, reassigned_labels)
            self.optimizer_con.zero_grad()
            con_loss.backward()
            self.optimizer_con.step()
            self.optimizer_ema.step()

            cross_entropy_loss = self.cross_entropy_loss(
                logit[:, sum(camera_person_list[:ith]):sum(camera_person_list[:ith + 1])],
                reassigned_labels)
            optimizer_cro.zero_grad()
            cross_entropy_loss.backward()
            optimizer_cro.step()
            self.optimizer_ema.step()
        self.writer.add_scalar("Contrastive loss", con_loss.item(), global_step=epoch)
        self.writer.add_scalar("Cross entropy loss", cross_entropy_loss.item(), global_step=epoch)

    def train_across_camera_view(self, train_dataloader, dream_dataloader, epoch, ith, optimizer_cci):
        train_iter = iter(train_dataloader)
        dream_iter = iter(dream_dataloader)

        len_train_iter = len(train_dataloader)
        len_dream_iter = len(dream_dataloader)

        for s in range(max(len_train_iter, len_dream_iter)+2):
            # +2 ensure the accessment of all data
            try:
                (images_tr_1, images_tr_2), _, _ = train_iter.__next__()
                assert images_tr_1.size(0) == self.opt.batch_size
            except(StopIteration, AssertionError):
                train_iter = iter(train_dataloader)
                (images_tr_1, images_tr_2), _, _ = train_iter.__next__()
            try:
                (images_dr_1, images_dr_2), _ = dream_iter.__next__()
                assert images_dr_1.size(0) == self.opt.batch_size
            except(StopIteration, AssertionError):
                dream_iter = iter(dream_dataloader)
                (images_dr_1, images_dr_2), _ = dream_iter.__next__()

            if images_tr_1.size(0) != self.opt.batch_size or images_dr_1.size(0) != self.opt.batch_size:
                continue

            images = images_tr_1.cuda()
            bsz = images.size(0)
            _, logit, x = self.model(images)
            # set CVC learning loss
            a = torch.exp(-1*torch.cdist(x, self.feat_buffer)/(self.opt.sigma**2)).detach()
            _, idx = torch.topk(a, self.opt.nearest, dim=1)
            b = torch.zeros_like(a).cuda()
            for i in range(bsz):
                b[i, idx[i, :]] = 1
            w = F.normalize(a * b, p=1, dim=1)
            logit_max, _ = torch.max(logit, dim=1, keepdim=True)
            logit = logit - logit_max.detach()
            cvc_loss = torch.trace(-1 * w @ F.log_softmax(logit, dim=1).t())
            self.writer.add_scalar("Cross-view correlation learning loss", cvc_loss.item(), global_step=epoch)

            if torch.cuda.is_available():
                images_tr_1, images_tr_2 = images_tr_1.cuda(), images_tr_2.cuda()
                images_dr_1, images_dr_2 = images_dr_1.cuda(), images_dr_2.cuda()

            with torch.no_grad():
                _, outputs_tr_1, _ = self.model(images_tr_1)
                _, outputs_tr_2, _ = self.model(images_tr_2)
                _, outputs_dr_1, _ = self.model(images_dr_1)
                _, outputs_dr_2, _ = self.model(images_dr_2)

                targets_tr = get_targets(self.opt, outputs_tr_1, outputs_tr_2)
                targets_dr = get_targets(self.opt, outputs_dr_1, outputs_dr_2)

            all_inputs = torch.cat([images_tr_1, images_tr_2, images_dr_1, images_dr_2], dim=0)
            all_targets = torch.cat([targets_tr, targets_tr, targets_dr, targets_dr], dim=0)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = (input_a + input_b) / 2
            mixed_target = (target_a + target_b) / 2

            # interleave original and dream images between batches
            # to get correct batch norm calculation
            mixed_input = list(torch.split(mixed_input, bsz))
            mixed_input = interleave(mixed_input, bsz)

            logits = [self.model(mixed_input[0])[1]]
            for inp in mixed_input[1:]:
                logits.append(self.model(inp)[1])

            # put interleaved samples back
            logits = interleave(logits, bsz)
            logits = torch.cat(logits, dim=0)

            mix_loss = self.mix_loss(logits, mixed_target)
            self.writer.add_scalar("Mix loss", mix_loss.item(), global_step=epoch)

            loss = mix_loss + self.opt.lamb * linear_rampup(ith, self.len_continual_index_list) * cvc_loss
            optimizer_cci.zero_grad()
            loss.backward()
            optimizer_cci.step()
            self.optimizer_ema.step()
