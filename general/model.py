import math
import torch
import torch.nn as nn
import einops as eo
import timm
from torchdyn.core import NeuralODE
from helper import cam_to_world, world_to_cam
from general.physics import get_exact_deqnet, get_analytic_physics

FPS = 15


class BasePyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = 0.1
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.LeakyReLU()
        self.depthrelu = nn.Softplus(beta=2, threshold=16)
        self.min_depth, self.max_depth = 2, 12

    def apply_norm(self, norm, x):
        B, C, H, W = x.shape
        x = eo.rearrange(x, 'b c h w -> b h w c')
        x = norm(x)
        x = eo.rearrange(x, 'b h w c -> b c h w')
        return x

    def coord_from_heatmap(self, x):
        B, C, H, W = x.shape

        x = eo.rearrange(x, 'b c h w -> b c (h w)')
        x = self.softmax(x / self.temperature)
        x = eo.rearrange(x, 'b c (h w) -> b c h w', h=H, w=W)

        x_ind = torch.tensor([i for i in range(W)], device=x.device)
        y_ind = torch.tensor([i for i in range(H)], device=x.device)
        grid_y, grid_x = torch.meshgrid(y_ind, x_ind, indexing='ij')

        result_x = torch.einsum('b c h w, h w -> b c h w', x, grid_x)
        result_y = torch.einsum('b c h w, h w -> b c h w', x, grid_y)

        x_coord = torch.sum(result_x, dim=(1, 2, 3)).unsqueeze(1)
        y_coord = torch.sum(result_y, dim=(1, 2, 3)).unsqueeze(1)

        coords = torch.cat((y_coord, x_coord), dim=1)

        return coords

    def depth_from_heatmap(self, x, z):
        if self.depth_mode == 'depthmap':
            assert x.shape == z.shape
            B, C, H, W = x.shape
            x = eo.rearrange(x, 'b c h w -> b c (h w)')
            x = self.softmax(x / self.temperature)
            x = eo.rearrange(x, 'b c (h w) -> b c h w', h=H, w=W)
            depth_map = torch.einsum('b c h w, b c h w -> b c h w', x, z)
            depth = torch.sum(depth_map, dim=(1, 2, 3))
            return depth
        else:
            z = z.mean(axis=(-1, -2))
            z = self.depth_head4(z).squeeze(-1)
            z = (self.max_depth - self.min_depth) * torch.nn.functional.sigmoid(z) + self.min_depth
            return z

    def get_coords3D(self, x, z):
        coords = self.coord_from_heatmap(x)
        depth = self.depth_from_heatmap(x, z)
        coords = torch.cat((coords, depth.unsqueeze(1)), dim=1)
        return coords


class ResNetPyramid(BasePyramid):
    def __init__(self, depth_mode='depthmap', backbone='resnet34'):
        super(ResNetPyramid, self).__init__()
        assert depth_mode in ['depthmap', 'regression']
        self.depth_mode = depth_mode
        num_stages = 4
        if backbone == 'resnet34':
            out_channels = [64, 64, 128, 256]
        elif backbone == 'resnet50':
            out_channels = [64, 256, 512, 1024]
        elif backbone == 'convnext':
            out_channels = [80, 160, 320, 640]
        elif backbone == 'hrnet':
            out_channels = [64, 128, 256, 512]
        elif backbone == 'convnextv2':
            out_channels = [80, 160, 320, 640]
        out_c = sum(out_channels[:num_stages])
        self.num_stages = num_stages
        self.pyramid = self.get_backbone(backbone)

        self.prehead = nn.Conv2d(out_c, out_c//2, (1, 1))
        self.head1 = nn.Conv2d(out_c//2, out_c//2, (7, 7), padding='same')
        nn.init.xavier_uniform_(self.head1.weight, math.sqrt(2))
        self.head2 = nn.Conv2d(out_c//2, out_c//4, (7, 7), padding='same')
        nn.init.xavier_uniform_(self.head2.weight, math.sqrt(2))
        self.head3 = nn.Conv2d(out_c // 4, 1, (1, 1))
        nn.init.xavier_uniform_(self.head3.weight)
        self.norm1 = nn.LayerNorm(out_c//2)
        self.norm2 = nn.LayerNorm(out_c//4)

        if self.depth_mode == 'depthmap':
            self.dprehead = nn.Conv2d(out_c, out_c // 2, (1, 1))
            self.dhead1 = nn.Conv2d(out_c // 2, out_c // 2, (7, 7), padding='same')
            nn.init.xavier_uniform_(self.dhead1.weight, math.sqrt(2))
            self.dhead2 = nn.Conv2d(out_c // 2, out_c // 4, (7, 7), padding='same')
            nn.init.xavier_uniform_(self.head2.weight, math.sqrt(2))
            self.dhead3 = nn.Conv2d(out_c // 4, 1, (1, 1))
            nn.init.xavier_uniform_(self.dhead3.weight)
            self.dnorm1 = nn.LayerNorm(out_c // 2)
            self.dnorm2 = nn.LayerNorm(out_c // 4)
        else:
            self.depth_head1 = nn.Conv2d(out_c, 256, (7, 7), stride=2)
            nn.init.xavier_uniform_(self.depth_head1.weight, math.sqrt(2))
            self.depth_head2 = nn.Conv2d(256, 256, (7, 7), stride=2)
            nn.init.xavier_uniform_(self.depth_head2.weight, math.sqrt(2))
            self.depth_head3 = nn.Conv2d(256, 512, (7, 7), stride=2)
            nn.init.xavier_uniform_(self.depth_head3.weight)
            self.depth_head4 = nn.Linear(512, 1)
            nn.init.xavier_uniform_(self.depth_head4.weight)
            self.depth_norm1 = nn.LayerNorm(256)
            self.depth_norm2 = nn.LayerNorm(256)
            self.depth_norm3 = nn.LayerNorm(512)

    def forward(self, x):
        B, C, H, W = x.shape
        feature_list = self.pyramid(x)

        aggregated_features = list()

        Hf, Wf = feature_list[0].shape[-2], feature_list[0].shape[-1]
        aggregated_features.append(feature_list[0])
        for feature in feature_list[1:self.num_stages]:
            feature = torch.nn.functional.interpolate(feature, size=(Hf, Wf), mode='bilinear')
            aggregated_features.append(feature)
        aggregated_features = torch.cat(aggregated_features, dim=-3)

        x = self.prehead(aggregated_features)
        x = self.relu(self.apply_norm(self.norm1, self.head1(x)))
        x = self.relu(self.apply_norm(self.norm2, self.head2(x)))
        x = self.head3(x)
        x = torch.nn.functional.interpolate(x, size=(H, W), mode='bilinear')

        if self.depth_mode == 'depthmap':
            z = self.dprehead(aggregated_features)
            z = self.relu(self.apply_norm(self.dnorm1, self.dhead1(z)))
            z = self.relu(self.apply_norm(self.dnorm2, self.dhead2(z)))
            z = self.dhead3(z)
            z = self.depthrelu(z)
            z = torch.nn.functional.interpolate(z, size=(H, W), mode='bilinear')
        else:
            x = torch.nn.functional.interpolate(x, size=(H, W), mode='bilinear')
            z = self.relu(self.apply_norm(self.depth_norm1, self.depth_head1(aggregated_features)))
            z = self.relu(self.apply_norm(self.depth_norm2, self.depth_head2(z)))
            z = self.depthrelu(self.apply_norm(self.depth_norm3, self.depth_head3(z)))

        return x, z

    def get_backbone(self, backbone_name):
        assert backbone_name in ['resnet34', 'resnet50', 'convnext', 'hrnet', 'convnextv2']
        if backbone_name in ['resnet34', 'resnet']:
            return timm.create_model('resnet34', features_only=True, pretrained=True, output_stride=32,
                                     out_indices=[i for i in range(self.num_stages)])
        elif backbone_name == 'resnet50':
            return timm.create_model('resnet50', features_only=True, pretrained=True, output_stride=32,
                                     out_indices=[i for i in range(self.num_stages)])
        elif backbone_name == 'convnext':
            return timm.create_model('convnext_nano', features_only=True, pretrained=True, output_stride=32,
                              out_indices=[i for i in range(self.num_stages)])
        elif backbone_name == 'hrnet':
            return timm.create_model('hrnet_w18', features_only=True, pretrained=True)
        elif backbone_name == 'convnextv2':
            return timm.create_model('convnextv2_nano', pretrained=True, features_only=True)


class ForecastNetwork(nn.Module):
    def __init__(self, environment_name='parcour', device='cpu', mode='exactNDE'):
        super(ForecastNetwork, self).__init__()
        assert mode in ['exactNDE']
        if 'parcour' in environment_name:
            self.torchdynnets = [NeuralODE(
                get_exact_deqnet(device, environment_name, xml_num=xml_num).to(device),
                sensitivity='autograd',
                solver='dopri5'
            ).to(device) for xml_num in range(3)]
            self.torchdynnets = nn.ModuleList(self.torchdynnets)
        else:
            dynnet = get_exact_deqnet(device, environment_name).to(device)
            self.torchdynnets = [NeuralODE(dynnet, sensitivity='autograd', solver='dopri5').to(device)]
            self.torchdynnets = nn.ModuleList(self.torchdynnets)
        self.simulation_timesteps_factor = 2
        self.device = device
        if environment_name in ['real_ball', 'realball']:
            self.FPS = 60
        else:
            self.FPS = FPS

    def forward(self, coords, timestamps, forecast_length = 10, extrinsic_matrix = None, xml_num=0):
        B, T, D = coords.shape
        assert T == 3 and D == 3

        timestamps = timestamps - timestamps[:, 1:2]

        # transform camara coordinates into reference coordinate system (coordinate system of center ball)
        if extrinsic_matrix is not None:
            coords = cam_to_world(coords, extrinsic_matrix)

        timestep = 1 / self.FPS

        virtual_timestep = timestep / self.simulation_timesteps_factor
        v = (coords[:, 2] - coords[:, 0]) / (timestamps[:, 2] - timestamps[:, 0]).unsqueeze(-1)
        r = coords[:, 1]
        x = torch.cat((r, v), dim=-1)

        timestamps = timestamps[:, 1:]
        # check how many frames are dropped
        num_dropped = (timestamps.shape[-1] - ((timestamps[:, 1:] - timestamps[:, :-1]) * self.FPS).sum(
            dim=-1)).max().round()

        #output_timesteps = torch.arange(0, 1, 1 / forecast_length).to(x.device)
        #t_span = torch.arange(0, 1 + 1e-4, 1 / forecast_length / self.simulation_timesteps_factor).to(x.device)
        t_span = torch.arange(0, (forecast_length + num_dropped) * timestep + 1e-4, virtual_timestep).to(self.device)

        t_eval, pred_r = self.torchdynnets[xml_num](x, t_span)
        pred_r = eo.rearrange(pred_r, 't b d -> b t d')

        distance_matrix = torch.cdist(t_eval[None, :, None], timestamps[:, :, None], p=1)
        indices = distance_matrix.argmin(dim=-2)

        #get predictions only where I have an image
        pred_r = torch.cat([pred_r[b:b + 1, indices[b], :3] for b in range(B)])

        # transform center coordinate system back to camera coordinate system
        if extrinsic_matrix is not None:
            pred_r = world_to_cam(pred_r, extrinsic_matrix)

        return pred_r


class AnalyticForecastNetwork(nn.Module):
    def __init__(self, environment_name='parcour', device='cpu'):
        super(AnalyticForecastNetwork, self).__init__()
        self.analyticPhysics = get_analytic_physics(device, environment_name)
        self.device = device
        self.FPS = FPS

    def forward(self, coords, timestamps, forecast_length = 10, extrinsic_matrix = None, xml_num=None):
        B, T_, D = coords.shape
        assert T_ == 3 and D == 3

        #transform camara coordinates into reference coordinate system (coordinate system of center ball)
        if extrinsic_matrix is not None:
            extrinsic_matrix = extrinsic_matrix.to(self.device)
            coords = cam_to_world(coords, extrinsic_matrix)

        timestamps = timestamps - timestamps[:, 1:2]

        v_0 = (coords[:, 2] - coords[:, 0]) / (timestamps[:, 2] - timestamps[:, 0]).unsqueeze(-1)
        r_0 = coords[:, 1]

        r_T = self.analyticPhysics(r_0, v_0, timestamps[:, 1:])

        #transform center coordinate system back to camera coordinate system
        if extrinsic_matrix is not None:
            r_T = world_to_cam(r_T, extrinsic_matrix)
        return r_T




def get_PEN(name='resnet34', depth_mode='depthmap', environment_name=None):
    assert name in ['resnet34', 'resnet50', 'convnext', 'hrnet', 'convnextv2']
    if name in ['resnet34', 'resnet50', 'convnext', 'hrnet', 'convnextv2']:
        model = ResNetPyramid(depth_mode=depth_mode, backbone=name)
    else:
        if depth_mode != 'depthmap':
            raise NotImplementedError
        else:
            raise ValueError

    if environment_name in ['realball']:
        model.min_depth = 0.05
        model.max_depth = 3
    elif environment_name in ['parcour_singleenv_singlecam', 'parcour_singleenv_multicam', 'parcour_multienv_singlecam',
                              'parcour_multienv_multicam', 'falling', 'carousel', 'parcour_dualenv_multicam']:
        model.min_depth = 2
        model.max_depth = 12
    else:
        raise ValueError

    return model


def get_PAF(device, mode='exactNDE', environment_name='parcour'):
    assert mode in ['exactNDE', 'analytic']
    assert environment_name in ['falling', 'carousel', 'parcour_singleenv_singlecam', 'parcour_singleenv_multicam',
                                'parcour_multienv_singlecam', 'parcour_multienv_multicam', 'realball',
                                'parcour_dualenv_multicam'
                                ]
    if mode == 'analytic' and environment_name not in ['falling', 'carousel']:
        raise NotImplementedError
    if mode == 'exactNDE':
        return ForecastNetwork(environment_name, device=device, mode=mode)
    elif mode == 'analytic':
        return AnalyticForecastNetwork(environment_name, device=device)
    else:
        raise NotImplementedError

