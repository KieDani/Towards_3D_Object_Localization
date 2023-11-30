import os
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='which GPU is to be used (if multiple GPUs are available)')
    parser.add_argument('--path', type=str, default=None, help='path to the saved models. If None, then the provided checkpoints from the paper are evaluated')
    args = parser.parse_args()
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
import torch
import numpy as np
import einops as eo
import matplotlib.pyplot as plt
import io
import cv2
import random
from tqdm import tqdm
import json
from helper import img_to_cam, cam_to_world, load_models, seed_worker, get_logs_path
from general.model import get_PEN
from general.config import EvalConfig
import general.dataset as my_dataset
from torch.utils.tensorboard import SummaryWriter
from general.transforms import val_transform, denorm
from paths import checkpoint_path as ch_pa

if __name__ == '__main__':
    my_dataset.video_length = 90


#img: (C, H, W); heatmap: (H, W)
def plot_heatmap(img, heatmap, title=''):
    fig = plt.figure()
    img = eo.rearrange(img, 'c h w -> h w c').cpu().numpy()
    heatmap = heatmap.cpu().numpy()
    plt.title(title)
    plt.imshow(img.astype(np.uint8))
    plt.imshow(heatmap.astype(np.uint8), cmap=plt.cm.viridis, alpha=0.65)
    plt.close()

#coords: (T, D); groundtruth: (T, D); limits = (min_lim, max_lim)
def plot3d(coords, groundtruth, limits, title=''):
    min_lim, max_lim = limits[0], limits[1]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], marker='o', label='estimated')
    ax.scatter(groundtruth[:, 0], groundtruth[:, 1], groundtruth[:, 2], marker='^', label='groundtruth')
    ax.set_xlabel('x', fontsize=18)
    ax.set_ylabel('y', fontsize=18)
    ax.set_zlabel('z', fontsize=18)
    ax.set_xlim3d(min_lim[0].item(), max_lim[0].item())
    ax.set_ylim3d(min_lim[1].item(), max_lim[1].item())
    ax.set_zlim3d(min_lim[2].item(), max_lim[2].item())
    ax.set_title(title, fontsize=20)
    plt.legend(bbox_to_anchor=(0.7, 1.02), loc="upper left", fontsize=16)
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    fig.clear()
    plt.close()

    return im


#coords: (T, D); groundtruth: (T, D)
def plot2d(coords, groundtruth, limits, title=''):
    min_lim, max_lim, x_lim = limits[0], limits[1], limits[2]
    fig = plt.figure()
    plt.title(title, fontsize=20)
    plt.ylabel(r'camera depth $z^{({C})}$', fontsize=18)
    plt.xlabel(r'frame index $n$', fontsize=18)
    plt.ylim(min_lim - 0.1*np.abs(max_lim - min_lim), max_lim + 0.1*np.abs(max_lim - min_lim))
    plt.xlim(0, x_lim)
    plt.plot(groundtruth[:, 2], color='red', label='ground truth')
    plt.plot(coords[:, 2], color='blue', label='estimated depth')
    plt.legend(fontsize=16)
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    plt.close()
    return im


def create_validationVideos(checkpoint_path, save_path, config, device='cpu', max_iter=-1):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    g = torch.Generator()
    g.manual_seed(0)

    checkpoint = load_models(None, None, checkpoint_path, device)
    coord_model_state_dict = checkpoint['coord_model_state_dict']
    coord_model = get_PEN(config.sin_title, environment_name=config.environment_name)
    coord_model.load_state_dict(coord_model_state_dict)
    coord_model = coord_model.to(device)
    coord_model.eval()

    if config.environment_name == 'realball':
        testset = my_dataset.get_dataset(config.environment_name, mode='test', transforms=val_transform, video_len=40)
    else:
        testset = my_dataset.get_dataset(config.environment_name, mode='test', transforms=val_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=1,
                                            worker_init_fn=seed_worker, generator=g)
    original_size = testloader.dataset.original_size

    num_cameras = testloader.sampler.data_source.num_cameras
    number_per_cam = {c: 0 for c in range(num_cameras)}
    with torch.no_grad():
        for i, data_dict in enumerate(tqdm(testloader)):
            for xml_num, stuff in data_dict.items():
                video, vidlabel, vidheatmap, eM, iM, d3label, cam_num, timestamps = stuff
                cam_num = cam_num.item()
                if max_iter> 0 and number_per_cam[cam_num] >= max_iter:
                    continue
                number_per_cam[cam_num] += 1

                video, d3label = video.to(device), d3label.to(device)
                iM, eM = iM.to(device), eM.to(device)
                video, d3label = video[0], d3label[0]
                heatmap, depthmap = coord_model(video)
                coords = coord_model.get_coords3D(heatmap, depthmap)

                T, C, H, W = video.shape
                coords_c = img_to_cam(coords, iM[0], original_size, (H, W))
                coords_w = cam_to_world(coords_c, eM.to(device))
                d3label_w = cam_to_world(d3label, eM[0].to(device))

                min_lim = torch.minimum(coords_c.min(0)[0], d3label.min(0)[0])
                max_lim = torch.maximum(coords_c.max(0)[0], d3label.max(0)[0])
                min_lim_w = torch.minimum(coords_w.min(0)[0], d3label_w.min(0)[0])
                max_lim_w = torch.maximum(coords_w.max(0)[0], d3label_w.max(0)[0])
                coords_c, d3label = coords_c.cpu().numpy(), d3label.cpu().numpy()
                coords_w, d3label_w = coords_w.cpu().numpy(), d3label_w.cpu().numpy()

                plots_2d = []
                for t in range(coords_c.shape[0]):
                    im = plot2d(coords_c[:t + 1], d3label[:t + 1],
                                (min_lim[2].cpu().numpy(), max_lim[2].cpu().numpy(), coords_c.shape[0]),
                                title='camera coordinate depth')
                    plots_2d.append(im)
                plots2d = np.asarray(plots_2d)

                plots_3d_c = []
                for t in range(coords_c.shape[0]):
                    # TODO check if (x, y) or (y, x) should be used
                    im_c = plot3d(coords_c[:t + 1, [1, 0, 2]], d3label[:t + 1][:, [1, 0, 2]],
                                  (min_lim[[1, 0, 2]].cpu().numpy(), max_lim[[1, 0, 2]].cpu().numpy()),
                                  title='camera coordinates')
                    plots_3d_c.append(im_c)
                plots_3d_c = np.asarray(plots_3d_c)

                plots_3d_w = []
                for t in range(coords_w.shape[0]):
                    im_w = plot3d(coords_w[:t + 1, :], d3label_w[:t + 1, :],
                                  (min_lim_w.cpu().numpy(), max_lim_w.cpu().numpy()), title='world coordinates')
                    plots_3d_w.append(im_w)
                plots_3d_w = np.asarray(plots_3d_w)

                save = os.path.join(save_path, f'xmlnum{xml_num:2}', f'camnum{cam_num:2}')
                os.makedirs(save, exist_ok=True)
                vid_num = len([e for e in os.listdir(save) if e.endswith('.mp4')])
                save = os.path.join(save, f'evalVid{vid_num:04d}.mp4')

                for t in range(video.shape[0]):
                    video[t] = denorm({'image': video[t]})['image']

                create_video(video.cpu().numpy(), heatmap.cpu().numpy(), depthmap.cpu().numpy(), plots2d, plots_3d_c,
                             plots_3d_w, title=save)




#images: (T, C, H, W); heatmaps: (T, H, W); depthmaps: (T, H, W); plots: (T, C, H, W)
def create_video(images, heatmaps, depthmaps, plots2d, plots3d_c, plots3d_w, title='output_bounce.mp4', imsave=True):
    T, C, H, W = images.shape
    images = eo.rearrange(images, 't c h w -> t h w c')
    fps = 3
    out = cv2.VideoWriter(title, cv2.VideoWriter_fourcc(*'mp4v'), fps, (3 * W, 2 * H))
    im_path = title.strip('.mp4')
    os.makedirs(im_path, exist_ok=True)

    for t in range(T):
        img = np.uint8(images[t])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if imsave: save_image(img, f'img{t:04d}.png', im_path)
        heat = np.uint8(heatmaps[t])
        heat = eo.rearrange(heat, 'c h w -> h w c')
        heat = np.uint8((heat - heatmaps.min()) * (255 / (heatmaps.max() - heatmaps.min())))
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        if imsave: save_image(heat, f'heat{t:04d}.png', im_path)
        depth = depthmaps[t]
        if len(depth.shape) > 3: #Filter if depth was regression output instead of depth_map
            depth = np.zeros_like(heat)
        else:
            depth = eo.rearrange(depth, 'c h w -> h w c')
            depth = np.uint8((depth - depthmaps.min()) * (255 / (depthmaps.max() - depthmaps.min()) + 1e-7))
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        if imsave: save_image(depth, f'depth{t:04d}.png', im_path)
        plot2d = np.uint8(plots2d[t])
        plot2d = cv2.cvtColor(plot2d, cv2.COLOR_BGRA2BGR)
        if imsave: save_image(plot2d, f'plot2d{t:04d}.png', im_path, size=(1024, 1024))
        plot2d = cv2.resize(plot2d, (W, H), interpolation=cv2.INTER_AREA)
        plot3d_c = np.uint8(plots3d_c[t])
        plot3d_c = cv2.cvtColor(plot3d_c, cv2.COLOR_BGRA2BGR)
        if imsave: save_image(plot3d_c, f'plot3d_c{t:04d}.png', im_path, size=(1024, 1024))
        plot3d_c = cv2.resize(plot3d_c, (W, H), interpolation=cv2.INTER_AREA)
        plot3d_w = np.uint8(plots3d_w[t])
        plot3d_w = cv2.cvtColor(plot3d_w, cv2.COLOR_BGRA2BGR)
        if imsave: save_image(plot3d_w, f'plot3d_w{t:04d}.png', im_path, size=(1024, 1024))
        plot3d_w = cv2.resize(plot3d_w, (W, H), interpolation=cv2.INTER_AREA)

        frame = np.hstack((np.vstack((img, heat)), np.vstack((depth, plot2d)), np.vstack((plot3d_c, plot3d_w))))
        out.write(frame)

    out.release()


def save_image(image, name, path, size=(512, 512)):
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(path, name), image)


def calc_metrics(checkpoint_path, config, device='cpu'):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    g = torch.Generator()
    g.manual_seed(0)

    num_bins = 3
    min_depth, max_depth = (0.05, 1.5) if config.environment_name == 'realball' else (4., 10.)

    checkpoint = load_models(None, None, checkpoint_path, device)
    coord_model_state_dict = checkpoint['coord_model_state_dict']
    coord_model = get_PEN(config.sin_title, environment_name=config.environment_name)
    coord_model.load_state_dict(coord_model_state_dict)
    coord_model = coord_model.to(device)
    coord_model.eval()

    testset = my_dataset.get_dataset(config.environment_name, mode='test', transforms=val_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=8,
                                            worker_init_fn=seed_worker, generator=g)
    original_size = testset.original_size

    with torch.no_grad():
        num_cameras = testloader.sampler.data_source.num_cameras
        num_environments = testloader.sampler.data_source.num_environments

        with torch.no_grad():
            distances = {i:{j:[] for j in range(num_cameras)} for i in range(num_environments)}
            number = {i:{j:0 for j in range(num_cameras)} for i in range(num_environments)}
            distances_bins = {i:{j:{k:[] for k in range(num_bins)} for j in range(num_cameras)} for i in range(num_environments)}
            number_bins = {i:{j:{k:0 for k in range(num_bins)} for j in range(num_cameras)} for i in range(num_environments)}
            for i, data_dict in enumerate(tqdm(testloader)):
                for xml_num, stuff in data_dict.items():
                    video, vidlabel, vidheatmap, eM, iM, d3label, cam_num, timestamps = stuff
                    images_all, d3labels_all = video.to(device), d3label.to(device)
                    eM, iM = eM.to(device), iM.to(device)

                    __, T, __, H, W = images_all.shape

                    images_all = eo.rearrange(images_all, 'b t c h w -> (b t) c h w')
                    heatmap_all, depthmap_all = coord_model(images_all)
                    coords_all = coord_model.get_coords3D(heatmap_all, depthmap_all)
                    coords_all = eo.rearrange(coords_all, '(b t) d -> b t d', t=T)
                    coords_all = img_to_cam(coords_all, iM, original_size, (H, W))
                    distance = (coords_all - d3labels_all).double().pow(2).sum(2).sqrt()

                    for cn in range(cam_num.shape[0]):
                        cn_ind = cam_num[cn].item()
                        for t in range(distance.shape[1]):
                            distances[xml_num][cn_ind].append(distance[cn, t].item())
                            number[xml_num][cn_ind] += 1
                            bin_width = (max_depth - min_depth) / num_bins
                            for b in range(num_bins):
                                if min_depth + bin_width * b <= d3labels_all[cn, t, 2].item() < min_depth + bin_width * (b + 1):
                                    distances_bins[xml_num][cn_ind][b].append(distance[cn, t].item())
                                    number_bins[xml_num][cn_ind][b] += 1

            merge = lambda x: [item for sublist in x for item in sublist]

            # mean and std per camera
            dtgs_cam = {xml_num:{cn:np.mean(distances[xml_num][cn]) for cn in range(num_cameras)} for xml_num in range(num_environments)}
            dtgs_cam_std = {xml_num:{cn:np.std(distances[xml_num][cn]) for cn in range(num_cameras)} for xml_num in range(num_environments)}

            dtgs_cam_bin = {xml_num:{cn:[np.mean(distances_bins[xml_num][cn][b]) for b in range(num_bins)] for cn in range(num_cameras)} for xml_num in range(num_environments)}
            dtgs_cam_bin_std = {xml_num:{cn:[np.std(distances_bins[xml_num][cn][b]) for b in range(num_bins)] for cn in range(num_cameras)} for xml_num in range(num_environments)}

            # mean and std per environment
            dtgs_env = {xml_num:np.mean(merge([distances[xml_num][cn] for cn in range(num_cameras)]))
                        for xml_num in range(num_environments)}
            dtgs_env_std = {xml_num:np.std(merge([distances[xml_num][cn] for cn in range(num_cameras)]))
                            for xml_num in range(num_environments)}

            dtgs_env_bin = {xml_num:[np.mean(merge([distances_bins[xml_num][cn][b] for cn in range(num_cameras)]))
                                     for b in range(num_bins)] for xml_num in range(num_environments)}
            dtgs_env_bin_std = {xml_num:[np.std(merge([distances_bins[xml_num][cn][b] for cn in range(num_cameras)]))
                                         for b in range(num_bins)] for xml_num in range(num_environments)}

            # mean and std for the whole dataset
            dtg = np.mean(merge([distances[xml_num][cn] for xml_num in range(num_environments) for cn in range(num_cameras)]))
            dtg_std = np.std(merge([distances[xml_num][cn] for xml_num in range(num_environments) for cn in range(num_cameras)]))

            dtg_bin = [np.mean(merge([distances_bins[xml_num][cn][b] for xml_num in range(num_environments) for cn in range(num_cameras)])) for b in range(num_bins)]
            dtg_bin_std = [np.std(merge([distances_bins[xml_num][cn][b] for xml_num in range(num_environments) for cn in range(num_cameras)])) for b in range(num_bins)]

            print('dtg: ', dtg)
            print('dtg_std: ', dtg_std)
            print('dtg_bin: ', dtg_bin)
            print('dtg_bin_std: ', dtg_bin_std)
            print('dtgs_env: ', dtgs_env)
            print('dtgs_env_std: ', dtgs_env_std)
            print('dtgs_env_bin: ', dtgs_env_bin)
            print('dtgs_env_bin_std: ', dtgs_env_bin_std)
            print('dtgs_cam: ', dtgs_cam)
            print('dtgs_cam_std: ', dtgs_cam_std)
            print('dtgs_cam_bin: ', dtgs_cam_bin)
            print('dtgs_cam_bin_std: ', dtgs_cam_bin_std)
            print('number: ', number)
            print('number_bins: ', number_bins)

            return {
                'dtg': dtg,
                'dtg_std': dtg_std,
                'dtg_bin': dtg_bin,
                'dtg_bin_std': dtg_bin_std,
                'dtgs_env': dtgs_env,
                'dtgs_env_std': dtgs_env_std,
                'dtgs_env_bin': dtgs_env_bin,
                'dtgs_env_bin_std': dtgs_env_bin_std,
                'dtgs_cam': dtgs_cam,
                'dtgs_cam_std': dtgs_cam_std,
                'dtgs_cam_bin': dtgs_cam_bin,
                'dtgs_cam_bin_std': dtgs_cam_bin_std,
                'number': number,
                'number_bins': number_bins
            }


def get_best_model(path, device='cuda:0'):
    model_names = [p for p in sorted(os.listdir(path)) if p.endswith('.pth')]
    best_dtg, best_model = 1e7, None
    for model_name in model_names:
        save_dict = torch.load(os.path.join(path, model_name), map_location=device)
        dtg = save_dict['DtG']
        #print(rdtg)
        best_dtg, best_model = (dtg, model_name) if dtg < best_dtg else (best_dtg, best_model)
    if best_model is None:
        raise ValueError('No model found')
    #print(best_rdtg)
    return best_model


def eval_paper():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    checkpoint_paths = [
        os.path.join(ch_pa, 'RD'),
        os.path.join(ch_pa, 'SD-S'),
        os.path.join(ch_pa, 'SD-M'),
        os.path.join(ch_pa, 'SD-L')
    ]
    for checkpoint_path in checkpoint_paths:
        checkpoint_path = os.path.join(checkpoint_path, get_best_model(checkpoint_path, device))
        path_parts = os.path.normpath(checkpoint_path).split(os.sep)
        sin_title = 'resnet34'
        environment = path_parts[-2]
        if environment == 'RD': environment = 'realball'
        elif environment == 'SD-S': environment = 'parcour_singleenv_singlecam'
        elif environment == 'SD-M': environment = 'parcour_singleenv_multicam'
        elif environment == 'SD-L': environment = 'parcour_multienv_multicam'
        folder = path_parts[-2]
        #TODO: save results of calc_metrics
        save_path = os.path.join(get_logs_path(), 'eval', folder, environment)
        print(f'evaluating {checkpoint_path}, saving to {save_path}')
        config = EvalConfig(sin_title, environment, '', folder)
        metrics = calc_metrics(checkpoint_path, config, device)
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        print('-'*50)
        #create_validationVideos(checkpoint_path, save_path, config, device, max_iter=10)


def eval_sup_mat():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    checkpoint_paths1 = [
        # Lossmodes TODO fill in Paths
        ]
    checkpoint_paths2 = [
        # Backbones TODO fill in Paths
        ]
    checkpoint_paths3 = [
        # Seed TODO fill in Paths
        ]
    checkpoint_paths4 = [
        # Analytic solution + Arbitrary physics TODO fill in Paths
        ]
    checkpoint_paths = [('lossmodes', checkpoint_paths1), ('backbones', checkpoint_paths2), ('seeds', checkpoint_paths3), ('arbitraryphysics', checkpoint_paths4)]
    for mode, checkpoints in checkpoint_paths:
        for checkpoint_path in checkpoints:
            checkpoint_path = os.path.join(checkpoint_path, get_best_model(checkpoint_path, device))
            path_parts = os.path.normpath(checkpoint_path).split(os.sep)
            identifier = path_parts[-2]
            sin_title = [i for i in identifier.split('_') if 'sintitle' in i][0].replace('sintitle:', '')
            environment = path_parts[-3]
            folder = path_parts[-4]
            save_path = os.path.join(get_logs_path(), 'eval_tmp', f'sup_mat_{mode}', environment, identifier)
            print(folder, environment, identifier)
            config = EvalConfig(sin_title, environment, identifier, folder)
            metrics = calc_metrics(checkpoint_path, config, device)
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
            print('-' * 50)
            # create_validationVideos(checkpoint_path, save_path, config, device, max_iter=10)

def eval_single_model(path):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = os.path.join(path, get_best_model(path, device))
    path_parts = os.path.normpath(checkpoint_path).split(os.sep)
    identifier = path_parts[-2]
    sin_title = [i for i in identifier.split('_') if 'sintitle' in i][0].replace('sintitle:', '')
    environment = path_parts[-3]
    if environment == 'RD': environment = 'realball'
    elif environment == 'SD-S': environment = 'parcour_singleenv_singlecam'
    elif environment == 'SD-M': environment = 'parcour_singleenv_multicam'
    elif environment == 'SD-L': environment = 'parcour_multienv_multicam'
    folder = path_parts[-4]
    save_path = os.path.join(get_logs_path(), 'eval', folder, environment, identifier)
    print(f'evaluating {checkpoint_path}, saving to {save_path}')
    config = EvalConfig(sin_title, environment, identifier, folder)
    metrics = calc_metrics(checkpoint_path, config, device)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print('-' * 50)
    # create_validationVideos(checkpoint_path, save_path, config, device, max_iter=10)


def main():
    if args.path is None:
        eval_paper()
    else:
        eval_single_model(args.path)



if __name__ == '__main__':
    main()