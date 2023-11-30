import torch
import torch.nn as nn
import xml.etree.ElementTree as ET
import math
from torch.func import vmap, grad

object_size = 0.0 # radius of moving object

class CarouselDE(nn.Module):
    def __init__(self, device='cpu'):
        super(CarouselDE, self).__init__()
        k = 1.73 * torch.ones((1,), dtype=torch.float32, device=device, requires_grad=True)
        self.k = nn.Parameter(k, requires_grad=True)
        self.center = torch.tensor([0, 0, 0], dtype=torch.float32, device=device, requires_grad=False)

    def forward(self, x, *args, **kwargs):
        r, v = x[:, :3], x[:, 3:]
        dv = - self.k**2 * (r - self.center)
        dr = v
        return torch.cat((dr, dv), dim=-1)


class BallDE(nn.Module):
    def __init__(self, device='cpu'):
        super(BallDE, self).__init__()
        g = 1 * torch.ones((1, ), dtype=torch.float32, device=device, requires_grad=True)
        self.g = nn.Parameter(g, requires_grad=True)
        self.c = 100 #Hight of potential wall
        self.temp = 0.05 #temperature to approximate heaviside with sigmoid
        self.device = device

    def forward(self, x, *args, **kwargs):
        r, v = x[:, :3], x[:, 3:]
        G = self.g * torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device).unsqueeze(0)
        dv = - G * (torch.sigmoid((r - torch.ones_like(r) * object_size) / self.temp) - \
                    self.c * torch.sigmoid(-(r - torch.ones_like(r) * object_size) / self.temp))
        dr = v
        return torch.cat((dr, dv), dim=-1)


class ParcourDE(nn.Module):
    def __init__(self, device='cpu', fastderivative=True, xml_num=0):
        super(ParcourDE, self).__init__()
        g = 1 * torch.ones((1, ), dtype=torch.float32, device=device, requires_grad=True)
        self.g = nn.Parameter(g, requires_grad=True)
        self.c = 1000 #Height of potential wall
        self.temp = 0.02 #temperature to approximate heaviside with sigmoid
        self.device = device
        self.environment = parse_xml(XMLs[xml_num])
        self.fastderivative = fastderivative
        self.dV = self.fast_dV if fastderivative else vmap(grad(self.potential))
        # needed to save the rotation matrix in order for it to be accessible outside of world_to_objectcoords()
        self.R = None

    def world_to_objectcoords(self, coords, objectdict):
        alpha = objectdict['angle']
        objectpos = torch.tensor(objectdict['pos'], dtype=torch.float32, device=coords.device)
        R = torch.tensor([[math.cos(math.radians(alpha)), math.sin(math.radians(alpha)), 0],
                          [-math.sin(math.radians(alpha)), math.cos(math.radians(alpha)), 0],
                          [0, 0, 1]], dtype=torch.float32, device=coords.device)
        self.R = R
        o_coords = coords - objectpos
        o_coords = torch.einsum('i j, ... j -> ... i', R, o_coords)
        return o_coords

    #You have to use vmap!
    def potential(self, coords):
        V = torch.zeros_like(coords[2])
        for objectdict in self.environment:
            o_r = self.world_to_objectcoords(coords, objectdict)
            o_size = torch.tensor(objectdict['size'], dtype=torch.float32, device=coords.device)
            o_size = o_size + torch.ones_like(o_size) * object_size
            # x, y, z = o_r[0], o_r[1], o_r[2]
            # a_x, a_y, a_z = o_size[0], o_size[1], o_size[2]
            # V += self.c * self.temp * \
            #       (torch.sigmoid((x + a_x) / self.temp) - torch.sigmoid((x - a_x) / self.temp)) * \
            #       (torch.sigmoid((y + a_y) / self.temp) - torch.sigmoid((y - a_y) / self.temp)) * \
            #       (torch.sigmoid((z + a_z) / self.temp) - torch.sigmoid((z - a_z) / self.temp))
            tmp = torch.sigmoid((o_r + o_size) / self.temp) - torch.sigmoid((o_r - o_size) / self.temp)
            V += self.c * self.temp * tmp[0] * tmp[1] * tmp[2]
        return V

    def fast_dV(self, coords):
        dV = torch.zeros_like(coords)
        for objectdict in self.environment:
            o_r = self.world_to_objectcoords(coords, objectdict)
            R = self.R
            o_size = torch.tensor(objectdict['size'], dtype=torch.float32, device=coords.device)
            o_size = o_size + torch.ones_like(o_size) * object_size
            dV_o = - self.c * \
                 (torch.sigmoid((o_r - o_size) / self.temp) - torch.sigmoid((o_r + o_size) / self.temp)).prod(dim=-1).unsqueeze(-1) \
                 * (torch.ones_like(coords) - torch.sigmoid((o_r - o_size) / self.temp) - torch.sigmoid((o_r + o_size) / self.temp))
            dV += torch.einsum(' i j, b i -> b j', R, dV_o)
        return dV

    def forward(self, t, x, *args, **kwargs):
        r, v = x[:, :3], x[:, 3:]
        e_z = torch.tensor([0, 0, 1], dtype=torch.float32, device=r.device)
        dv = - self.g * e_z.unsqueeze(0) * torch.sigmoid((r-torch.ones_like(r) * object_size) / self.temp) + \
             self.c * e_z.unsqueeze(0) * torch.sigmoid(-(r - torch.ones_like(r) * object_size) / self.temp)
        dv += - self.dV(r)
        dr = v
        return torch.cat((dr, dv), dim=-1)


class BallAnalytic(nn.Module):
    def __init__(self, device):
        super(BallAnalytic, self).__init__()
        g = 1. * torch.ones((1,), dtype=torch.float32, device=device, requires_grad=True)
        self.g = nn.Parameter(g, requires_grad=True)
        self.device = device

    def forward(self, r_0, v_0, timestamps):
        B, D = r_0.shape
        r_T = torch.empty((B, timestamps.shape[1], D), device=self.device)
        G = self.g * torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device).unsqueeze(0)

        h = r_0[:, 2] - object_size * torch.ones_like(r_0[:, 2])
        vB = torch.sqrt(v_0[:, 2] ** 2 + 2 * self.g * torch.max(h, torch.zeros_like(h)))
        tB = (v_0[:, 2] + vB) / self.g
        tB_2 = 2 * vB / self.g
        t_lastbounce = torch.empty((B, timestamps.shape[1], 1), dtype=torch.float32, device=self.device)
        t_lastbounce[:, 0, 0] = tB
        for i in range(0, timestamps.shape[1]):
            ti = timestamps[:, i]
            t_lastbounce[:, i, 0] = torch.where(ti > t_lastbounce[:, i - 1, 0] + tB_2, t_lastbounce[:, i - 1, 0] + tB_2,
                                                t_lastbounce[:, i - 1, 0])

        r_T[:, :, :2] = v_0[:, :2].unsqueeze(1) * timestamps.unsqueeze(-1) + r_0[:, :2].unsqueeze(1)
        r_T[:, :, 2] = torch.where(timestamps < t_lastbounce.squeeze(-1),
                                   - 0.5 * G[:, 2].unsqueeze(1) * timestamps ** 2 + v_0[:, 2].unsqueeze(
                                       1) * timestamps + r_0[:, 2].unsqueeze(1),
                                   - 0.5 * G[:, 2].unsqueeze(1) * (timestamps - t_lastbounce.squeeze(-1)) ** 2 + vB.unsqueeze(
                                       1) * (timestamps - t_lastbounce.squeeze(-1)))
        return r_T


class CarouselAnalytic(nn.Module):
    def __init__(self, device):
        super(CarouselAnalytic, self).__init__()
        k = 1.73 * torch.ones((1,), dtype=torch.float32, device=device, requires_grad=True)
        self.k = nn.Parameter(k, requires_grad=True)
        self.center = torch.tensor([0, 0, 0], dtype=torch.float32, device=device, requires_grad=False)
        self.device = device

    def forward(self, r_0, v_0, timestamps):
        B, D = r_0.shape  # T+1 Values
        r_T = 1. / self.k * v_0.unsqueeze(1) * torch.sin(self.k * timestamps.unsqueeze(-1)) + (r_0 - self.center).unsqueeze(1) * torch.cos(self.k * timestamps.unsqueeze(-1)) + self.center
        return r_T


class RealBallDE(nn.Module):
    def __init__(self, device='cpu', fastderivative=True):
        super(RealBallDE, self).__init__()
        g = 9.81 * torch.ones((1, ), dtype=torch.float32, device=device, requires_grad=True)
        self.g = nn.Parameter(g, requires_grad=True)
        self.c = 1000 #Height of potential wall
        self.temp1 = 0.005 #temperature to approximate heaviside with sigmoid for the walls
        self.temp2 = 0.015 #temperature to approximate heaviside with sigmoid for the objects
        self.device = device
        self.environment = parse_xml(XML_realball)
        assert fastderivative == True, "Only fast derivative is implemented"
        self.fastderivative = fastderivative
        self.dV = self.fast_dV # if fastderivative else vmap(grad(self.potential))
        # needed to save the rotation matrix in order for it to be accessible outside of world_to_objectcoords()
        self.R = None
        self.object_size = 0.

    def world_to_objectcoords(self, coords, objectdict):
        alpha = objectdict['angle']
        objectpos = torch.tensor(objectdict['pos'], dtype=torch.float32, device=coords.device)
        R = torch.tensor([[math.cos(math.radians(alpha)), math.sin(math.radians(alpha)), 0],
                          [-math.sin(math.radians(alpha)), math.cos(math.radians(alpha)), 0],
                          [0, 0, 1]], dtype=torch.float32, device=coords.device)
        self.R = R
        o_coords = coords - objectpos
        o_coords = torch.einsum('i j, ... j -> ... i', R, o_coords)
        return o_coords

    def fast_dV(self, coords):
        dV = torch.zeros_like(coords)
        for objectdict in self.environment:
            o_r = self.world_to_objectcoords(coords, objectdict)
            R = self.R
            o_size = torch.tensor(objectdict['size'], dtype=torch.float32, device=coords.device)
            o_size = o_size + torch.ones_like(o_size) * self.object_size
            dV_o = - self.c * \
                 (torch.sigmoid((o_r - o_size) / self.temp2) - torch.sigmoid((o_r + o_size) / self.temp2)).prod(dim=-1).unsqueeze(-1) \
                 * (torch.ones_like(coords) - torch.sigmoid((o_r - o_size) / self.temp2) - torch.sigmoid((o_r + o_size) / self.temp2))
            dV += torch.einsum(' i j, b i -> b j', R, dV_o)
        return dV

    def forward(self, t, x, *args, **kwargs):
        r, v = x[:, :3], x[:, 3:]
        ones = torch.ones_like(r)
        e_z = torch.tensor([0, 0, 1], dtype=torch.float32, device=r.device)
        e_x = torch.tensor([1, 0, 0], dtype=torch.float32, device=r.device)
        e_y = torch.tensor([0, 1, 0], dtype=torch.float32, device=r.device)
        dv = - self.g * e_z.unsqueeze(0) * torch.sigmoid((r - ones * self.object_size) / self.temp1) + \
            self.c * e_z.unsqueeze(0) * torch.sigmoid(-(r - ones * self.object_size) / self.temp1) + \
            self.c * e_x.unsqueeze(0) * torch.sigmoid(-(r - ones * self.object_size) / self.temp1) + \
            - self.c * e_y.unsqueeze(0) * torch.sigmoid((r - ones * 0.6 + ones * self.object_size) / self.temp1)
        # TODO: adjust position of the walls above
        dv += - self.dV(r)
        dr = v
        return torch.cat((dr, dv), dim=-1)



def parse_xml(xml_file):
    objectlist = []
    root = ET.fromstring(xml_file)
    for parent in root:
        if parent.tag == 'worldbody':
            for child in parent:
                if child.tag == 'body' and child.attrib['name'] != 'freeball':
                    objectdict = {
                        'pos': [float(c) for c in child.attrib['pos'].split(' ')],
                        'angle': float(child.attrib['euler'].split(' ')[-1]) if 'euler' in child.attrib else 0.
                    }
                    for grandchild in child:
                        if grandchild.tag == 'geom':
                            objectdict['size'] = [float(g) for g in grandchild.attrib['size'].split(' ')]
                    objectlist.append(objectdict)
                    #TODO get size too
    return objectlist

XMLs = []
XML0 = r"""
<mujoco>
  <option gravity="0 0 -0.0875">
  </option>
  <asset>
    <material name="floor" reflectance=".2"/>
  </asset>
  <worldbody>
        <light diffuse=".5 .5 .5" pos="-2 -2 10" dir="0.1 0.1 -1"/>
        <geom type="plane" size="800 800 0.1" rgba="0.6 0.6 0.6 1" pos="0 0 0" material="floor"/>
    <body pos="1 0 0" name="freeball">
      <joint type="free" name="j"/>
      <geom type="sphere" size=".2" mass="1.0" rgba="0.0 0.32 0.34 1" name="g" solref="-1000 0"/>
    </body>
    <body pos="2.2 2.2 0" name="obstacle1" euler="0 0 45">
      <geom name="o1" type="box" size=".8 1.4 1.3" rgba="0.34 0.76 1. 1" material="floor"/>
    </body>
    <body pos="-0.5 -0.5 0" name="obstacle2">
      <geom name="o2" type="box" size="1.0 .7 .4" rgba="0.99 0.83 0.16 1" material="floor"/>
    </body>
    <body pos="2.5 -2 0" name="obstacle3" euler="0 0 -45">
      <geom name="o3" type="box" size="0.5 2.0 2.2" rgba="0.61 0.85 0.69 1" material="floor"/>
    </body>
    <body pos="-3.0 0.5 0" name="obstacle4" euler="0 0 -45">
      <geom name="o4" type="box" size="0.8 .8 .8" rgba="0.61 0.85 0.69 1" material="floor"/>
    </body>
  </worldbody>
</mujoco>
"""
XMLs.append(XML0)


# exchange the colors of the obstacles; change positions and sizes
XML1 = r"""
<mujoco>
    <option gravity="0 0 -0.0875">
    </option>
    <asset>
        <material name="floor" reflectance=".2"/>
    </asset>
    <worldbody>
        <light diffuse=".5 .5 .5" pos="-2 -2 10" dir="0.1 0.1 -1"/>
        <geom type="plane" size="800 800 0.1" rgba="0.6 0.6 0.6 1" pos="0 0 0" material="floor"/>
        <body pos="1 0 0" name="freeball">
            <joint type="free" name="j"/>
            <geom type="sphere" size=".2" mass="1.0" rgba="0.0 0.32 0.34 1" name="g" solref="-1000 0"/>
        </body>
        <body pos="-0.5 -2.5 0" name="obstacle1" euler="0 0 45">
      <geom name="o1" type="box" size=".8 1.3 0.9" rgba="0.61 0.85 0.69 1" material="floor"/>
    </body>
    <body pos="-1.5 +1.8 0" name="obstacle2">
      <geom name="o2" type="box" size="0.9 .8 .7" rgba="0.34 0.76 1. 1" material="floor"/>
    </body>
    <body pos="2 1.5 0" name="obstacle3" euler="0 0 45">
      <geom name="o3" type="box" size="0.5 1.8 3.0" rgba="0.99 0.83 0.16 1" material="floor"/>
    </body>
  </worldbody>
</mujoco>
"""
XMLs.append(XML1)


XML2 = r"""
<mujoco>
    <option gravity="0 0 -0.0875">
    </option>
    <asset>
        <material name="floor" reflectance=".2"/>
    </asset>
    <worldbody>
        <light diffuse=".5 .5 .5" pos="-2 -2 10" dir="0.1 0.1 -1"/>
        <geom type="plane" size="800 800 0.1" rgba="0.6 0.6 0.6 1" pos="0 0 0" material="floor"/>
        <body pos="1 0 0" name="freeball">
            <joint type="free" name="j"/>
            <geom type="sphere" size=".2" mass="1.0" rgba="0.0 0.32 0.34 1" name="g" solref="-1000 0"/>
        </body>
    <body pos="1.7 1.5 0" name="obstacle1" euler="0 0 45">
      <geom name="o1" type="box" size="0.5 3.8 2.5" rgba="0.99 0.83 0.16 1" material="floor"/>
    </body>
    <body pos="2.5 0.5 0" name="obstacle2" euler="0 0 -45">
      <geom name="o2" type="box" size="0.5 4.2 1.5" rgba="0.99 0.83 0.56 0.7" material="floor"/>
    </body>
    <body pos="-1.5 -0.5 0" name="obstacle3" euler="0 0 45">
        <geom name="o3" type="box" size="0.5 0.8 0.3" rgba="0.34 0.76 1. 1" material="floor"/>
    </body>
  </worldbody>
</mujoco>
"""
XMLs.append(XML2)


XML_realball = r"""
<mujoco>
    <worldbody>
        <body pos="0.4 0.3 0" name="obstacle1" euler="0 0 0">
          <geom name="o1" type="box" size="0.4 0.3 0.46"/>
        </body>
  </worldbody>
</mujoco>
"""


def get_exact_deqnet(device='cpu', environment_name='parcour', xml_num=0):
    if 'parcour' in environment_name:
        return ParcourDE(device=device, fastderivative=True, xml_num=xml_num)
    elif environment_name == 'carousel':
        return CarouselDE(device=device)
    elif environment_name == 'falling':
        return BallDE(device=device)
    elif environment_name == 'realball':
        return RealBallDE(device=device, fastderivative=True)
    else:
        raise NotImplementedError


def get_analytic_physics(device='cpu', environment_name='parcour'):
    if environment_name in ['parcour_singleenv_singlecam', 'parcour_singleenv_multi',
                            'parcour_multienv_singlecam', 'parcour_multienv_multi']:
        raise NotImplementedError
    elif environment_name == 'carousel':
        return CarouselAnalytic(device=device)
    elif environment_name == 'falling':
        return BallAnalytic(device=device)
    else:
        raise NotImplementedError