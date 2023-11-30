import numpy as np
import torch
import numpy
import random
import socket
import os
import torch.linalg as LA

from paths import logs_path, data_path, checkpoint_path


def get_logs_path():
    return logs_path


def get_data_path():
    return data_path



def create_heatmaps(joint, output_size=(480, 480), sigma=8, factor=1):
    '''
    create heatmap from keypoints x, y
    joint: (y, x)
    output_size: (height, width)
    '''
    gaus2d = lambda x, y: 100 * np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    y = np.arange(int(output_size[0] / factor))
    x = np.arange(int(output_size[1] / factor))
    X, Y = np.meshgrid(x, y)
    heatmap = np.zeros((1, int(output_size[0] / factor), int(output_size[1] / factor)), dtype=np.float32)
    y0, x0 = joint[0] / factor, joint[1] / factor
    heatmap[0] = gaus2d(Y - y0, X - x0)
    return heatmap

def resized_to_original_keypopints(keypoints, original_size, resized_size):
    '''
    resized keypoints to original size
    keypoints: (N, 2) with order (y, x)
    original_size: order (H, W)
    resized_size: order (H, W)
    '''
    keypoints[..., 1] = original_size[1] / resized_size[1] * (keypoints[..., 1] + 0.5) - 0.5
    keypoints[..., 0] = original_size[0] / resized_size[0] * (keypoints[..., 0] + 0.5) - 0.5
    return keypoints


def original_to_resized_keypopints(keypoints, original_size, resized_size):
    '''
    original keypoints to resized size
    keypoints: (N, 2) with order (y, x)
    original_size: order (H, W)
    resized_size: order (H, W)
    '''
    keypoints[..., 1] = resized_size[1] / original_size[1] * (keypoints[..., 1] + 0.5) - 0.5
    keypoints[..., 0] = resized_size[0] / original_size[0] * (keypoints[..., 0] + 0.5) - 0.5
    return keypoints



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def save_models(coord_model, forec_model, coord_optimizer, forec_optimizer, lossesandmetrics, epoch, identifier='', training_parameters = None, config=None):
    if config is not None:
        identifier = config.get_pathforsaving()
        training_parameters = config.__dict__

    logs_path = get_logs_path()
    save_path = os.path.join(logs_path, 'checkpoints', identifier)
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f'{epoch}.pth')

    torch.save({
        'epoch': epoch,
        'coord_model_state_dict': coord_model.state_dict(),
        'coord_optimizer_state_dict': coord_optimizer.state_dict(),
        'coord_loss': lossesandmetrics['loss_2Dcoord'],
        'forec_model_state_dict': forec_model.state_dict(),
        'forec_optimizer_state_dict': forec_optimizer.state_dict(),
        'forec_loss': lossesandmetrics['loss_reproject'],
        'training_parameters': training_parameters,
        'DtG': lossesandmetrics['distance_to_gt'],
        'rDtG': lossesandmetrics['rel_distance_to_gt'],
    }, save_path)

    return save_path


def load_models(identifier, epoch, path=None, device='cuda:0'):
    if path is None:
        assert identifier is None and epoch is None
        path = os.path.join(get_logs_path(), 'checkpoints', identifier, f'{epoch}.pth')

    checkpoint = torch.load(path, map_location=device)

    return checkpoint

def img_to_cam(coords, Mint, original_size, resized_size):
    '''
    coords: (..., 3) with ordering (y, x, z)
    Mint: (3, 3) or (B, 3, 3)
    original_size: order (height, width)
    resized_size: order (height, width)
    ----------------
    return: (..., 3) with ordering (y, x, z)
    '''
    coords_c = coords.clone()
    coords_c[..., :2] = resized_to_original_keypopints(coords_c[..., :2], original_size, resized_size)
    coords_c[..., [0, 1]] = coords_c[..., [1, 0]] * coords[..., 2:3]
    if len(Mint.shape) == 3:
        inv_Mint = torch.linalg.inv(Mint[:, :3, :3])
        coords_c = torch.einsum('b i d, b ... d -> b ... i', inv_Mint, coords_c)
    elif len(Mint.shape) == 2:
        inv_Mint = torch.linalg.inv(Mint[:3, :3])
        coords_c = torch.einsum('i d, ... d -> ... i', inv_Mint, coords_c)
    else:
        raise ValueError('Mint should be 2D or 3D tensor')
    coords_c[..., [0, 1, 2]] = coords_c[..., [1, 0, 2]]
    return coords_c


#flips from yxz format to xyz format
def cam_to_world(coords_c, extrinsic_matrix):
    #coords_c[..., [0, 1]] = coords_c[..., [1, 0]]
    tmp = coords_c[..., [1, 0, 2]]
    inverse_extrinsic_matrix = torch.linalg.inv(extrinsic_matrix)
    #coords_c = torch.cat((coords_c, torch.ones_like(coords_c[..., 0:1])), dim=-1)
    tmp = torch.cat((tmp, torch.ones_like(tmp[..., 0:1])), dim=-1)
    if len(tmp.shape) == 3: inverse_extrinsic_matrix = inverse_extrinsic_matrix.unsqueeze(-3)
    #coords_w = torch.einsum('i d, ... d -> ... i', inverse_extrinsic_matrix, coords_c)
    coords_w = torch.einsum('... i d, ... d -> ... i', inverse_extrinsic_matrix, tmp)
    coords_w = coords_w[..., :3] / coords_w[..., 3:4]
    return coords_w

#flips from xyz format to yxz format
def world_to_cam(coords_w, extrinsic_matrix):
    coords_w = torch.cat((coords_w, torch.ones_like(coords_w[..., 0:1])), dim=-1)
    if len(coords_w.shape) == 3: extrinsic_matrix = extrinsic_matrix.unsqueeze(-3)
    coords_c = torch.einsum('... i d, ... d -> ... i', extrinsic_matrix, coords_w)
    coords_c = coords_c[..., :3] / coords_c[..., 3:4]
    coords_c[..., [0, 1]] = coords_c[..., [1, 0]]
    return coords_c


def cam_to_img(coords_c, Mint, original_size, resized_size):
    '''
    coords_c: (..., 3) with ordering (y, x, z)
    Mint: (3, 3) or (B, 3, 3)
    original_size: order (height, width)
    resized_size: order (height, width)
    ----------------
    returns: (..., 3) with ordering (y, x, z)
    '''
    coords = coords_c.clone()
    coords[..., [0, 1]] = coords[..., [1, 0]]
    if len(Mint.shape) == 3:
        orig_Mint = Mint[:, :3, :3]
        coords = torch.einsum('b i d, b ... d -> b ... i', orig_Mint, coords)
    elif len(Mint.shape) == 2:
        orig_Mint = Mint[:3, :3]
        coords = torch.einsum('i d, ... d -> ... i', orig_Mint, coords)
    else:
        raise ValueError('Mint should be 2D or 3D tensor')
    coords[..., [0, 1]] = coords[..., [1, 0]] / coords[..., 2:3].clone()
    coords = original_to_resized_keypopints(coords, original_size, resized_size)
    return coords


def get_reprojectionloss(loss_coord_title):
    assert loss_coord_title in ['L1', 'scale_invariant', 'alpha']
    if loss_coord_title == 'L1':
        loss_fn_coord = torch.nn.L1Loss(reduction='none')
    elif loss_coord_title == 'scale_invariant':
        loss_fn_coord = lambda a, b: LA.norm((a - b), dim=-1) / (LA.norm(a, dim=-1) + LA.norm(b, dim=-1))
    else:
        loss_fn_coord = lambda a, b: torch.abs(LA.norm(a, dim=-1) - LA.norm(b, dim=-1)) / (
                    LA.norm(a, dim=-1) + LA.norm(b, dim=-1)) + 0.5 * (1 - torch.einsum('... d, ... d -> ...', a, b) / (
                    LA.norm(a, dim=-1) * LA.norm(b, dim=-1)))
    return loss_fn_coord


def update_ema(model, model_ema, alpha=0.95):
    with torch.no_grad():
        for name, param in model_ema.named_parameters():
            model.state_dict()[name]
            param.data = alpha * param + (1 - alpha) * model.state_dict()[name].data
        for name, param in model_ema.named_buffers():
            param.data = alpha * param + (1 - alpha) * model.state_dict()[name].data
        return model_ema



def map_environment_name(environment_name):
    mapper = {
        'parcour': 'parcour_singleenv_singlecam',
        'parcour_single': 'parcour_singleenv_singlecam',
        'parcour_singleenv_singlecam': 'parcour_singleenv_singlecam',
        'parcour_multi': 'parcour_singleenv_multicam',
        'parcour_singleenv_multicam': 'parcour_singleenv_multicam',
        'parcour_multienv_multi': 'parcour_multienv_multicam',
        'parcour_multienv_multicam': 'parcour_multienv_multicam',
        'parcour_multienv_single': 'parcour_multienv_singlecam',
        'parcour_multienv_singlecam': 'parcour_multienv_singlecam',
        'carousel': 'carousel',
        'falling': 'falling',
        'real_ball': 'realball',
        'realball': 'realball',
        'parcour_dualenv_multicam': 'parcour_dualenv_multicam',
    }
    return mapper[environment_name.lower()]


if __name__ == "__main__":
    a1 = create_heatmaps(np.asarray([100, 488, 1.]), output_size=(480, 480), sigma=2)
    pass