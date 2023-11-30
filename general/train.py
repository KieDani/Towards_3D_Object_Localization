import os
if __name__ == '__main__':
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--environment_name', type=str, default='parcour')
    parser.add_argument('--lossmode', type=str, default='2Dpred_nograd')
    parser.add_argument('--folder', type=str, default='tmp')
    parser.add_argument('--sin_title', type=str, default='convnext')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--loss_coord_title', type=str, default='L1')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
import torch
# if __name__ == '__main__':
#     torch.autograd.set_detect_anomaly(True)
from tqdm import tqdm
import random
import einops as eo
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import scipy
import cv2
import os

from helper import create_heatmaps, seed_worker, save_models, get_reprojectionloss
from helper import img_to_cam, cam_to_img, cam_to_world, update_ema
from general.model import get_PEN, get_PAF
from general.dataset import get_dataset
from general.evaluate import plot2d, plot3d
from general.config import MyConfig
from general.transforms import train_transform, val_transform, denorm


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
debug = False

def train(config):
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    g = torch.Generator()
    g.manual_seed(config.seed)

    forecast_length = max(config.timesteps)

    logs_path = config.get_logs_path(debug)

    writer = SummaryWriter(logs_path)

    coord_model = get_PEN(config.sin_title, config.depth_mode, config.environment_name).to(device)
    coord_model_ema = get_PEN(config.sin_title, config.depth_mode, config.environment_name).to(device).eval()
    coord_model_ema.load_state_dict(coord_model.state_dict())
    coord_model_ema = update_ema(coord_model, coord_model_ema, 0)

    if hasattr(coord_model, 'temperature'):
        coord_model.temperature = config.temperature
        coord_model_ema.temperature = config.temperature
    forec_model = get_PAF(device, config.forec_title, config.environment_name).to(device)

    trainset = get_dataset(config.environment_name, mode='train', transforms=train_transform)
    valset = get_dataset(config.environment_name, mode='val', transforms=val_transform)
    original_size = trainset.original_size

    num_workers = 0 if debug else 8
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.BATCH_SIZE, shuffle=True,
                                              num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    valloader = torch.utils.data.DataLoader(valset, batch_size=config.BATCH_SIZE//4, shuffle=False,
                                            num_workers=num_workers, worker_init_fn=seed_worker, generator=g)

    loss_fn = torch.nn.MSELoss(reduction='none')

    loss_fn_coord = get_reprojectionloss(config.loss_coord_title)

    optim1 = torch.optim.Adam(coord_model.parameters(), lr=config.lr_coord)
    optim2 = torch.optim.Adam(forec_model.parameters(), lr=config.lr_forec)

    loss_weight = lambda i: config.reprojection_weight * \
                            max(0, min(i - config.warmup_iterations, 400 + config.warmup_iterations)) \
                            / (400 + config.warmup_iterations)

    min_losses = [(1e7, -1), (1e7, -1), (1e7, -1), (1e7, -1), (1e7, -1)]
    iteration = 0
    for epoch in range(config.NUM_EPOCHS):
        print('Epoch', epoch)
        for i, data_dict in enumerate(tqdm(trainloader)):
            for xml_num, stuff in data_dict.items():
                optim1.zero_grad()
                optim2.zero_grad()
                video, vidlabel, vidheatmap, eM, iM, d3label, cam_num, timestamps = stuff
                t = random.randint(1, video.shape[1] - 2 - forecast_length)
                timestamps_T = timestamps[:, t-1:t+forecast_length+1]
                images_0 = video[:, t - 1:t + 2]
                t_steps = [t + ts for ts in config.timesteps]
                images_t = video[:, t_steps]
                imlabels = vidlabel[:, [t] + t_steps]
                imheatmaps = vidheatmap[:, t_steps]
                images_0, images_t, imlabels, imheatmaps = images_0.to(device), images_t.to(device), imlabels.to(device), imheatmaps.to(device)
                eM, iM = eM.to(device), iM.to(device)
                timestamps_T = timestamps_T.to(device)

                images_0 = eo.rearrange(images_0, 'b t c h w -> (b t) c h w')
                heatmap, depthmap = coord_model(images_0)
                coords = coord_model.get_coords3D(heatmap, depthmap)
                coords = eo.rearrange(coords, '(b t) d -> b t d', t=3)
                #heatmap_0 = eo.rearrange(heatmap, '(b t) c h w -> b t c h w', t=3)[:, 1:2]

                #f = iM[:, 1, 1]
                __, __, H, W = images_0.shape
                coords = img_to_cam(coords, iM, original_size, (H, W))

                pred_coords = forec_model(coords, timestamps=timestamps_T, forecast_length=forecast_length, extrinsic_matrix = eM.to(device), xml_num=xml_num)
                pred_coords_t = pred_coords[:, config.timesteps]
                if config.lossmode in ['2Dgt', '2Dpred', '2Dpred_nograd']:
                    pred_coords_t = cam_to_img(pred_coords_t, iM, original_size, (H, W))[..., :2]

                images_t = eo.rearrange(images_t, 'b t c h w -> (b t) c h w')
                heatmap_t, depthmap_t = coord_model(images_t)
                coords_t = coord_model.get_coords3D(heatmap_t, depthmap_t)
                coords_t = eo.rearrange(coords_t, '(b t) d -> b t d', t=len(config.timesteps))
                coords_c_t = img_to_cam(coords_t, iM, original_size, (H, W))
                heatmap_t = eo.rearrange(heatmap_t, '(b t) c h w -> b t c h w', t=len(config.timesteps))

                if config.lossmode == '2Dgt':
                    loss_reproject = loss_fn_coord(pred_coords_t, imlabels[:, 1:]).mean()
                elif config.lossmode == '2Dpred':
                    loss_reproject = loss_fn_coord(pred_coords_t, coords_t[..., :2]).mean()
                elif config.lossmode == '2Dpred_nograd':
                    loss_reproject = loss_fn_coord(pred_coords_t, coords_t[..., :2].detach()).mean()
                elif config.lossmode == '3Dgt':
                    d3gt = d3label[:, t_steps].to(device)
                    loss_reproject = loss_fn_coord(pred_coords_t, d3gt).mean()
                elif config.lossmode == '3Dpred':
                    loss_reproject = loss_fn_coord(pred_coords_t, coords_c_t).mean()
                elif config.lossmode == '3Dpred_nograd':
                    loss_reproject = loss_fn_coord(pred_coords_t, coords_c_t.detach()).mean()
                else:
                    raise ValueError('Unknown lossmode')

                # heatmap_t = eo.rearrange(heatmap_t, 'b t c h w -> b t c (h w)').softmax(dim=-1) * imheatmaps.sum(dim=[-1, -2]).unsqueeze(-1)
                # heatmap_t = eo.rearrange(heatmap_t, 'b t c (h w) -> b t c h w', h=H, w=W)
                loss_2Dheatmap = loss_fn(heatmap_t, imheatmaps).mean()

                loss = (loss_weight(iteration) * loss_reproject + loss_2Dheatmap) / len(data_dict.keys())
                loss.backward(retain_graph=False)

                torch.nn.utils.clip_grad_norm_(coord_model.parameters(), 5)
                torch.nn.utils.clip_grad_norm_(forec_model.parameters(), 5)
                if iteration % 2 == 0 or iteration < config.warmup_iterations:
                    optim1.step()
                if iteration % 2 == 1:# and iteration > config.warmup_iterations:
                    #TODO: remove comment, if forec_model should be trained, too
                    #optim2.step()
                    optim1.step()

                coord_model_ema = update_ema(coord_model, coord_model_ema, config.ema_decay)


            writer.add_scalar('Training/reprojection loss', loss_reproject, epoch*len(trainloader)+i)
            writer.add_scalar('Training/2D heatmap loss', loss_2Dheatmap, epoch * len(trainloader) + i)

        #__, loss_2Dcoord, loss_reproject, distance_to_gt, rel_distance_to_gt = validation((coord_model, forec_model), valloader, writer, epoch, device, config)
        __, loss_2Dcoord, loss_reproject, distance_to_gt, rel_distance_to_gt = validation((coord_model_ema, forec_model), valloader, writer, epoch, device, config)
        coord_model_ema.eval()

        # save model, if it is one of the three best ones
        min_loss5, epoch5 = min_losses[0]
        if rel_distance_to_gt < min_loss5 and debug == False:
            min_losses[0] = (rel_distance_to_gt, epoch)
            min_losses = sorted(min_losses, reverse=True)
            lossesandmetrics = {
                'loss_2Dcoord': loss_2Dcoord,
                'loss_reproject': loss_reproject,
                'distance_to_gt': distance_to_gt,
                'rel_distance_to_gt': rel_distance_to_gt
            }
            save_path = save_models(coord_model, forec_model, optim1, optim2, lossesandmetrics, epoch, config=config)
            save_path = os.path.dirname(save_path)
            if epoch > 4: os.remove(os.path.join(save_path, f'{epoch5}.pth'))

        iteration += 1


def validation(models, valloader, writer, epoch, device, config):
    coord_model, forec_model = models
    coord_model.eval()
    forec_model.eval()

    original_size = valloader.dataset.original_size
    forecast_length = 15
    timesteps = [i for i in range(1, forecast_length)]
    num_cameras = valloader.sampler.data_source.num_cameras
    num_environments = valloader.sampler.data_source.num_environments

    loss_fn = torch.nn.MSELoss(reduction='none')
    loss_fn_coord = get_reprojectionloss(config.loss_coord_title)
    with torch.no_grad():
        losses_metrics = {}
        number_metrics = {}
        for i in range(num_environments):
            losses_metrics[i] = {
                'loss_reproject': {},
                'loss_2Dheatmap': {},
                'loss_2Dcoord': {},
                'loss_z_c': {},
                'distance_to_gt': {},
                'rel_distance_to_gt': {},
            }
            number_metrics[i] = {}
            for j in range(num_cameras):
                for v in losses_metrics[i].values():
                    v[j] = 0
                number_metrics[i][j] = 0
        for i, data_dict in enumerate(tqdm(valloader)):
            for xml_num, stuff in data_dict.items():
                video, vidlabel, vidheatmap, eM, iM, d3label, cam_num, timestamps = stuff
                t = 1
                t_steps = [t + ts for ts in timesteps]
                images_0 = video[:, t - 1:t + 2]
                images_T = video[:, t_steps]
                timestamps_T = timestamps[:, t-1:t+forecast_length+1]
                images_all = video[:, t:t+forecast_length]
                d3labels_all = d3label[:, t:t+forecast_length]
                imlabels = vidlabel[:, [t] + t_steps]
                imheatmaps = vidheatmap[:, t_steps]
                images_0, images_T, imlabels, imheatmaps = images_0.to(device), images_T.to(device), imlabels.to(device), imheatmaps.to(device)
                images_all, d3labels_all = images_all.to(device), d3labels_all.to(device)
                eM, iM = eM.to(device), iM.to(device)
                timestamps_T = timestamps_T.to(device)

                images_0 = eo.rearrange(images_0, 'b t c h w -> (b t) c h w')
                heatmap, depthmap = coord_model(images_0)
                coords = coord_model.get_coords3D(heatmap, depthmap)
                coords = eo.rearrange(coords, '(b t) d -> b t d', t=3)
                #heatmap_0 = eo.rearrange(heatmap, '(b t) c h w -> b t c h w', t=3)[:, 1]
                coords_0 = coords[:, 1]

                images_T = eo.rearrange(images_T, 'b t c h w -> (b t) c h w')
                heatmap_T, depthmap_T = coord_model(images_T)
                coords_T = coord_model.get_coords3D(heatmap_T, depthmap_T)
                coords_T = eo.rearrange(coords_T, '(b t) d -> b t d', t=len(t_steps))
                images_T = eo.rearrange(images_T, '(b t) c h w -> b t c h w', t=len(t_steps))
                heatmap_T = eo.rearrange(heatmap_T, '(b t) c h w -> b t c h w', t=len(t_steps))

                #f = iM[:, 1, 1]
                __, __, H, W = images_0.shape
                coords_c = img_to_cam(coords, iM, original_size, (H, W))
                #pred_coords = forec_model(coords_c, forecast_length=forecast_length)
                pred_coords = forec_model(coords_c, timestamps=timestamps_T, forecast_length=forecast_length, extrinsic_matrix = eM.to(device), xml_num=xml_num)
                pred_coords_T = pred_coords[:, timesteps]

                images_all = eo.rearrange(images_all, 'b t c h w -> (b t) c h w')
                heatmap_all, depthmap_all = coord_model(images_all)
                coords_all = coord_model.get_coords3D(heatmap_all, depthmap_all)
                coords_all = eo.rearrange(coords_all, '(b t) d -> b t d', t=forecast_length)
                f = iM[:, 1, 1]
                __, __, H, W = images_all.shape
                coords_all = img_to_cam(coords_all, iM, original_size, (H, W))
                distance = (coords_all - d3labels_all).double().pow(2).sum(2).sqrt()
                rel_distance = distance / d3labels_all.double().pow(2).sum(2).sqrt()
                pred_coords_T_B = cam_to_img(pred_coords_T, iM, original_size, (H, W))[..., :2]
                loss_reproject_tmp = loss_fn_coord(pred_coords_T_B, coords_T[..., :2])
                loss_2Dcoord_tmp = loss_fn_coord(coords_0[:, :2], imlabels[:, 0]) * 1e-3
                # heatmap_T = eo.rearrange(heatmap_T, 'b t c h w -> b t c (h w)').softmax(dim=-1) * imheatmaps.sum(dim=[-1, -2]).unsqueeze(-1)
                # heatmap_T = eo.rearrange(heatmap_T, 'b t c (h w) -> b t c h w', h=H, w=W)
                loss_2Dheatmap_tmp = loss_fn(heatmap_T, imheatmaps)

                for cn_ind in range(cam_num.shape[0]):
                    cn = cam_num[cn_ind].item()
                    number_metrics[xml_num][cn] += 1
                    losses_metrics[xml_num]['distance_to_gt'][cn] += distance[cn_ind].mean().item()
                    losses_metrics[xml_num]['rel_distance_to_gt'][cn] += rel_distance[cn_ind].mean().item()
                    losses_metrics[xml_num]['loss_reproject'][cn] += loss_reproject_tmp[cn_ind].mean().cpu().item()
                    # maybe also add at time t
                    losses_metrics[xml_num]['loss_2Dcoord'][cn] += loss_2Dcoord_tmp[cn_ind].mean().cpu().item()
                    losses_metrics[xml_num]['loss_2Dheatmap'][cn] += loss_2Dheatmap_tmp[cn_ind].mean().cpu().item()

        distance_to_gt = np.mean([losses_metrics[xml_num]['distance_to_gt'][cn] / number_metrics[xml_num][cn] for cn in range(num_cameras) for xml_num in range(num_environments)])
        loss_2Dheatmap = np.mean([losses_metrics[xml_num]['loss_2Dheatmap'][cn] / number_metrics[xml_num][cn] for cn in range(num_cameras) for xml_num in range(num_environments)])
        loss_2Dcoord = np.mean([losses_metrics[xml_num]['loss_2Dcoord'][cn] / number_metrics[xml_num][cn] for cn in range(num_cameras) for xml_num in range(num_environments)])
        loss_reproject = np.mean([losses_metrics[xml_num]['loss_reproject'][cn] / number_metrics[xml_num][cn] for cn in range(num_cameras) for xml_num in range(num_environments)])
        rel_distance_to_gt = np.mean([losses_metrics[xml_num]['rel_distance_to_gt'][cn] / number_metrics[xml_num][cn] for cn in range(num_cameras) for xml_num in range(num_environments)])

        for xml_num in range(num_environments):
            for metric in losses_metrics[xml_num].keys():
                for cn in range(num_cameras):
                    losses_metrics[xml_num][metric][cn] /= number_metrics[xml_num][cn]

        for xmlnum in range(num_environments):
            for camnum in range(num_cameras):
                writer.add_scalar(f'Validation Loss/env{xmlnum}/camera{camnum}/reprojection loss', losses_metrics[xml_num]['loss_reproject'][camnum], epoch)
                writer.add_scalar(f'Validation Loss/env{xmlnum}/camera{camnum}/2D coordinate loss', losses_metrics[xml_num]['loss_2Dcoord'][camnum], epoch)
                writer.add_scalar(f'Validation Loss/env{xmlnum}/camera{camnum}/2D heatmap loss', losses_metrics[xml_num]['loss_2Dheatmap'][camnum], epoch)
                writer.add_scalar(f'Validation Metric/env{xmlnum}/camera{camnum}/Distance to groundtruth', losses_metrics[xml_num]['distance_to_gt'][camnum], epoch)
                writer.add_scalar(f'Validation relative Metric/env{xmlnum}/camera{camnum}/Relative distance to groundtruth', losses_metrics[xml_num]['rel_distance_to_gt'][camnum], epoch)

        writer.add_text(f'Validation/env{0}/camera{camnum}/coord_model t=0:', ''.join((str(coords_c[0, 1, 0].item()), ', ', str(coords_c[0, 1, 1].item()), ', ', str(coords_c[0, 1, 2].item()))), epoch)
        coords_T_c = img_to_cam(coords_T, iM, original_size, (H, W))[:, -1]
        writer.add_text(f'Validation/env{0}/camera{camnum}/coord_model t=T:', ''.join((str(coords_T_c[0, 1].item()), ', ', str(coords_T_c[0, 0].item()), ', ', str(coords_T_c[0, 2].item()))), epoch)
        writer.add_text(f'Validation/env{0}/camera{camnum}/forec_model t=T:', ''.join((str(pred_coords_T[0, -1, 0].item()), ', ', str(pred_coords_T[0, -1, 1].item()), ', ', str(pred_coords_T[0, -1, 2].item()))), epoch)
        writer.add_text(f'Validation/env{0}/camera{camnum}/ground truth t=0:', ''.join((str(d3label[0, t, 1].item()), ', ', str(d3label[0, t, 0].item()), ', ', str(d3label[0, t, 2].item()))), epoch)
        writer.add_text(f'Validation/env{0}/camera{camnum}/ground truth t=T:', ''.join((str(d3label[0, t + forecast_length - 1, 1].item()), ', ', str(d3label[0, t + forecast_length - 1, 0].item()), ', ', str(d3label[0, t + forecast_length - 1, 2].item()))), epoch)
        if hasattr(forec_model, 'analyticPhysics') and hasattr(forec_model.analyticPhysics, 'g'):
            writer.add_text('model parameters:', f'{forec_model.analyticPhysics.g.item()}', epoch)
        elif hasattr(forec_model, 'torchdynnets') and hasattr(forec_model.torchdynnets[0].vf.vf, 'g'):
            writer.add_text('model parameters:', f'{forec_model.torchdynnets[0].vf.vf.g.item()}', epoch)
        if hasattr(forec_model, 'analyticPhysics') and hasattr(forec_model.analyticPhysics, 'k'):
            writer.add_text('model parameters:', f'{forec_model.analyticPhysics.k.item()}', epoch)
        elif hasattr(forec_model, 'dynnet') and hasattr(forec_model.torchdynnets[0].vf.vf, 'k'):
            writer.add_text('model parameters:', f'{forec_model.torchdynnets[0].vf.vf.k.item()}', epoch)

        vmin = 0
        vmax = imheatmaps.max().item()

        fig = plt.figure()
        img = eo.rearrange(images_0, '(b t) c h w -> b t c h w', t=3)
        img = denorm({'image': img[0, 1]})['image'].cpu().numpy()
        img = eo.rearrange(img, 'c h w -> h w c')
        heatmap = eo.rearrange(heatmap, '(b t) c h w -> b t c h w', t=3)[0, 1].squeeze(0)
        heatmap = heatmap.cpu().numpy()
        plt.title(''.join((str(coords_0[0, 0].cpu().numpy().item()), ', ', str(coords_0[0, 1].cpu().numpy().item()))))
        plt.imshow(img.astype(np.uint8))
        plt.imshow(heatmap, cmap=plt.cm.viridis, alpha=0.65, vmin=heatmap.min(), vmax=heatmap.max())
        writer.add_figure('validation heatmap coord_model t=0', fig, epoch)
        plt.close()

        fig = plt.figure()
        img = eo.rearrange(images_0, '(b t) c h w -> b t c h w', t=3)
        img = denorm({'image': img[0, 1]})['image'].cpu().numpy()
        img = eo.rearrange(img, 'c h w -> h w c')
        gt_coord = imlabels[0, 0]
        H, W, __ = img.shape
        joint = (gt_coord[0].cpu().numpy().item(), gt_coord[1].cpu().numpy().item(), 1.0)
        heatmap = create_heatmaps(joint, (H, W)).squeeze(0)
        plt.title(''.join((str(joint[0]), ', ', str(joint[1]))))
        plt.imshow(img.astype(np.uint8))
        plt.imshow(heatmap, cmap=plt.cm.viridis, alpha=0.65, vmin=vmin, vmax=vmax)
        writer.add_figure('validation heatmap ground truth t=0', fig, epoch)
        plt.close()

        fig = plt.figure()
        img = denorm({'image': images_T[0, -1]})['image'].cpu().numpy()
        img = eo.rearrange(img, 'c h w -> h w c')
        heatmap = heatmap_T[0, -1].squeeze(0)
        heatmap = heatmap.cpu().numpy()
        plt.title(''.join((str(coords_T[0, -1, 0].cpu().numpy().item()), ', ', str(coords_T[0, -1, 1].cpu().numpy().item()))))
        plt.imshow(img.astype(np.uint8))
        plt.imshow(heatmap, cmap=plt.cm.viridis, alpha=0.65, vmin=heatmap.min(), vmax=heatmap.max())
        writer.add_figure(f'validation heatmap coord_model t={forecast_length - 1}', fig, epoch)
        plt.close()

        fig = plt.figure()
        img = denorm({'image': images_T[0, -1]})['image'].cpu().numpy()
        img = eo.rearrange(img, 'c h w -> h w c')
        H, W, __ = img.shape
        imgcoords = cam_to_img(pred_coords_T[0, -1], iM[0], original_size, (H, W))
        joint = (imgcoords[0].cpu().numpy().item(), imgcoords[1].cpu().numpy().item(), 1.)
        heatmap = create_heatmaps(joint, (H, W)).squeeze(0)
        plt.title(''.join((str(joint[0]), ', ', str(joint[1]))))
        plt.imshow(img.astype(np.uint8))
        plt.imshow(heatmap, cmap=plt.cm.viridis, alpha=0.65, vmin=vmin, vmax=vmax)
        writer.add_figure(f'validation heatmap forec_model t={forecast_length - 1}', fig, epoch)
        plt.close()

        fig = plt.figure()
        img = denorm({'image': video[0, t + 5]})['image'].cpu().numpy()
        img = eo.rearrange(img, 'c h w -> h w c')
        H, W, __ = img.shape
        imgcoords = cam_to_img(pred_coords[0, 5], iM[0], original_size, (H, W))
        joint = (imgcoords[0].cpu().numpy().item(), imgcoords[1].cpu().numpy().item(), 1.)
        heatmap = create_heatmaps(joint, (H, W)).squeeze(0)
        plt.title(''.join((str(joint[0]), ', ', str(joint[1]))))
        plt.imshow(img.astype(np.uint8))
        plt.imshow(heatmap, cmap=plt.cm.viridis, alpha=0.65, vmin=vmin, vmax=vmax)
        writer.add_figure('validation heatmap forec_model t=5', fig, epoch)
        plt.close()

        fig = plt.figure()
        img = denorm({'image': video[0, t + 8]})['image'].cpu().numpy()
        img = eo.rearrange(img, 'c h w -> h w c')
        H, W, __ = img.shape
        imgcoords = cam_to_img(pred_coords[0, 8], iM[0], original_size, (H, W))
        joint = (imgcoords[0].cpu().numpy().item(), imgcoords[1].cpu().numpy().item(), 1.)
        heatmap = create_heatmaps(joint, (H, W)).squeeze(0)
        plt.title(''.join((str(joint[0]), ', ', str(joint[1]))))
        plt.imshow(img.astype(np.uint8))
        plt.imshow(heatmap, cmap=plt.cm.viridis, alpha=0.65, vmin=vmin, vmax=vmax)
        writer.add_figure('validation heatmap forec_model t=8', fig, epoch)
        plt.close()

        fig = plt.figure()
        img = denorm({'image': video[0, t + 13]})['image'].cpu().numpy()
        img = eo.rearrange(img, 'c h w -> h w c')
        H, W, __ = img.shape
        imgcoords = cam_to_img(pred_coords[0, 13], iM[0], original_size, (H, W))
        joint = (imgcoords[0].cpu().numpy().item(), imgcoords[1].cpu().numpy().item(), 1.)
        heatmap = create_heatmaps(joint, (H, W)).squeeze(0)
        plt.title(''.join((str(joint[0]), ', ', str(joint[1]))))
        plt.imshow(img.astype(np.uint8))
        plt.imshow(heatmap, cmap=plt.cm.viridis, alpha=0.65, vmin=vmin, vmax=vmax)
        writer.add_figure('validation heatmap forec_model t=13', fig, epoch)
        plt.close()

        fig = plt.figure()
        img = eo.rearrange(images_0, '(b t) c h w -> b t c h w', t=3)
        img = denorm({'image': img[0, 1]})['image'].cpu().numpy()
        img = eo.rearrange(img, 'c h w -> h w c')
        H, W, __ = img.shape
        imgcoords = cam_to_img(pred_coords[0, 0], iM[0], original_size, (H, W))
        joint = (imgcoords[0].cpu().numpy().item(), imgcoords[1].cpu().numpy().item(), 1.)
        heatmap = create_heatmaps(joint, (H, W)).squeeze(0)
        plt.title(''.join((str(joint[0]), ', ', str(joint[1]))))
        plt.imshow(img.astype(np.uint8))
        plt.imshow(heatmap, cmap=plt.cm.viridis, alpha=0.65, vmin=vmin, vmax=vmax)
        writer.add_figure('validation heatmap forec_model t=0', fig, epoch)
        plt.close()

        # if epoch % 20 == 0:
        #     heatmaps, depthmaps = coord_model(video[-1, :].to(device))
        #     coords_c = coord_model.get_coords3D(heatmaps, depthmaps)
        #     coords_c = img_to_cam(coords_c, iM[-1], original_size, (H, W))
        #     imgs = torch.stack([denorm({'image': video[-1, t]})['image'] for t in range(video.shape[1])])
        #     validationvideo = create_video(images=imgs.cpu().numpy(),
        #                                    heatmaps=heatmaps.cpu().numpy(),
        #                                    depthmaps=depthmaps.cpu().numpy(),
        #                                    coords_c=coords_c.cpu().numpy(),
        #                                    d3labels=d3label[-1, :].cpu().numpy(),
        #                                    f=f[-1], eM=eM[-1].cpu())
        #     writer.add_video('validation video coord_model', validationvideo, epoch, fps=3)

    coord_model.train()
    forec_model.train()

    return loss_2Dheatmap, loss_2Dcoord, loss_reproject, distance_to_gt, rel_distance_to_gt


# images: (T C H W)
def create_video(images, heatmaps, depthmaps, coords_c, d3labels, f, eM):
    T, C, H, W = images.shape
    # max_depth = 12
    # min_depth = 2
    # depthmaps = np.where(depthmaps > max_depth, max_depth, depthmaps)
    # depthmaps = np.where(depthmaps < min_depth, min_depth, depthmaps)
    coords_w = np.array(cam_to_world(torch.tensor(coords_c), eM))
    d3label_w = np.array(cam_to_world(torch.tensor(d3labels), eM))
    min_lim = np.minimum(coords_c.min(0), d3labels.min(0))
    max_lim = np.maximum(coords_c.max(0), d3labels.max(0))
    min_lim_w = np.minimum(coords_w.min(0), d3label_w.min(0))
    max_lim_w = np.maximum(coords_w.max(0), d3label_w.max(0))

    plots_2d = []
    for t in range(coords_c.shape[0]):
        im = plot2d(coords_c[:t + 1], d3labels[:t + 1],
                    (min_lim[2], max_lim[2], coords_c.shape[0]), title='camera coordinate depth')
        im = scipy.ndimage.zoom(im, (H/im.shape[0], W/im.shape[1], 1), order=1)
        im = np.uint8(im)
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        plots_2d.append(im)
    plots2d = np.asarray(plots_2d)

    plots_3d_c = []
    for t in range(coords_c.shape[0]):
        # TODO check if (x, y) or (y, x) should be used
        im_c = plot3d(coords_c[:t + 1, [1, 0, 2]], d3labels[:t + 1, [1, 0, 2]],
                      (min_lim[[1, 0, 2]], max_lim[[1, 0, 2]]), title='camera coordinates')
        im_c = np.uint8(im_c)
        im_c = cv2.cvtColor(im_c, cv2.COLOR_BGRA2BGR)
        im_c = scipy.ndimage.zoom(im_c, (H / im_c.shape[0], W / im_c.shape[1], 1), order=1)
        plots_3d_c.append(im_c)
    plots_3d_c = np.asarray(plots_3d_c)

    plots_3d_w = []
    for t in range(coords_w.shape[0]):
        im_w = plot3d(coords_w[:t + 1, :], d3label_w[:t + 1, :],
                      (min_lim_w, max_lim_w), title='world coordinates')
        im_w = np.uint8(im_w)
        im_w = cv2.cvtColor(im_w, cv2.COLOR_BGRA2BGR)
        im_w = scipy.ndimage.zoom(im_w, (H / im_w.shape[0], W / im_w.shape[1], 1), order=1)
        plots_3d_w.append(im_w)
    plots_3d_w = np.asarray(plots_3d_w)

    heats = list()
    for t in range(heatmaps.shape[0]):
        heat = eo.rearrange(heatmaps[t], 'c h w -> h w c')
        heat = np.uint8((heat - heatmaps.min()) * (255 / (heatmaps.max() - heatmaps.min())))
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heats.append(heat)
    heats = np.asarray(heats)

    depths = list()
    for t in range(depthmaps.shape[0]):
        depth = eo.rearrange(depthmaps[t], 'c h w -> h w c')
        depth = np.uint8((depth - depthmaps.min()) * (255 / (depthmaps.max() - depthmaps.min() + 1e-7)))
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        depths.append(depth)
    depths = np.asarray(depths)

    images = np.uint8(eo.rearrange(images, 't c h w -> t h w c'))


    video = np.concatenate(
        (np.concatenate((images, depths, plots_3d_c), axis=2),
         np.concatenate((heats, plots2d, plots_3d_w), axis=2)), axis=1
    )[None]

    video = eo.rearrange(video, 'b t h w c -> b t c h w')

    return video




def main():
    global debug
    debug = args.debug
    timesteps = [2, 4, 6] if args.environment_name == 'realball' else [4, 8, 15]
    config = MyConfig(lr_coord=args.lr, timesteps=timesteps, loss_coord_title=args.loss_coord_title, forec_title='exactNDE', sin_title=args.sin_title, environment_name=args.environment_name, folder=args.folder, lossmode=args.lossmode)
    train(config)



if __name__ == '__main__':
    main()


