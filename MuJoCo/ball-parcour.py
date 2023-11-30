import mujoco
import numpy as np
import math
import PIL.Image
import random
import csv
import sys
import os
import argparse
import shutil
import xml.etree.ElementTree as ET
from xml_parcour import XMLs

# Troubleshooting notes(WSL):
# try running this command to make rendering work by creating an in-memory frame buffer
#   xvfb-run -a -s "-screen 0 1400x900x24" bash
# if problems persist, the following command might help but should not be necessary for MuJoCo 2.1.4 and higher (GLEW was replaced with GLAD)
#   export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

bounce = True

# General settings:
fps = 15  # frames per second
record_time = 6  # in seconds
mark_pos = False  # will mark the path of the small ball as it travels
max_width = 224  # screen width
max_height = 224  # screen height
dataset_size = 1000

# Camera settings:
camera_angles = [
    [40, -60, 8],
    [0, -65, 10],
    [60, -55, 10],
    [20, -60, 9],
    [50, -70, 7],
    [10, -50, 9],
    [30, -60, 10],
    [0, -70, 8],
    [60, -50, 8],
]

# give out helpful information when prompted
if len(sys.argv) >= 2 and (sys.argv[1] == 'help' or sys.argv[1] == '-h' or sys.argv[1] == '-help'):
    print(
        'This script will simulate 1 ball falling down for 6 seconds, as well as label data. Refer to the contents of the script for more details.')
    print('Usage: ball-carousel.py <input .csv> <output folder> <optional render origin flag>')
    print(
        'Input is optional, if not provided, ball position and velocity will be randomized. Inputting random allows for random data while also being able to specify an output folder.')
    print(
        'Output folder defaults to creating a folder named \'output\'. Output files are recorded frames in .png format and label information in .csv format.')
    print('Set render origin flag to 1 if the global coordinate system origin should be rendered with its axis.')
    print('When encountering errors related to rendering, consult \'Troubleshooting notes\' inside the script.')
    sys.exit()


def simulation(output, camera_mode, XML):
    # init mujoco + GLcontext
    # print("Mujoco Version " + mujoco.mj_versionString())

    gl = mujoco.GLContext(max_width, max_height)
    gl.make_current()

    model = mujoco.MjModel.from_xml_string(XML)

    #If the ball is supposed to bounce, gravity has to be increased
    if bounce:
        model.opt.gravity = np.array([0, 0, -1])
    else:
        model.opt.gravity = np.array([0, 0, -0.04])

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    scene = mujoco.MjvScene(model, maxgeom=100)
    cam = mujoco.MjvCamera()
    options = mujoco.MjvOption()

    #options.frame = mujoco.mjtFrame.mjFRAME_WORLD

    # camera settings
    # defines camera position via a spherical coordinate system, using lookat as the origin:
    # default camera settings
    cam.lookat[0] = 0
    cam.lookat[1] = 0
    cam.lookat[2] = 0
    cam.azimuth = camera_angles[camera_mode][0]
    cam.elevation = camera_angles[camera_mode][1]
    cam.distance = camera_angles[camera_mode][2]

    # init graphics related stuff
    mujoco.mjv_updateScene(model, data, options, mujoco.MjvPerturb(), cam, mujoco.mjtCatBit.mjCAT_ALL, scene);
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, ctx)
    viewport = mujoco.MjrRect(0, 0, max_width, max_height)

    # init ball with data
    # if no data was read, generate random data with some simple constraints
    # to make sure the ball does actually cross through the image

    # -2.5 < x < 1.5, -2.5 < y < 1.5, 1 < z < 4
    r = np.random.rand(3) * np.array([4., 4., 3.]) + np.array([-2.5, -2.5, 1])
    data.joint('j').qpos[0] = r[0]  # random.uniform(-2, 2)
    data.joint('j').qpos[1] = r[1]  # random.uniform(-2, 2)
    data.joint('j').qpos[2] = r[2]  # random.uniform(-2, 2)

    # set initial velocity v

    # 0.5 < v_xy < 1; 0.5 < v_xy < 1; -0.5 < v_z < 0.5 ; noise is added additionally between -0.25 and 0.25
    xy_center = np.array([0., 0.])
    e_xy = xy_center - r[:2] / np.linalg.norm(xy_center - r[:2])
    v = np.zeros(3, dtype=np.float64)
    v[:2] = (np.random.rand() * 0.5 + 0.5) * e_xy
    v[2] = np.random.rand() * 1. - 0.5
    v += np.random.rand(3) * 0.5 - 0.25
    data.joint('j').qvel[0] = v[0]  # random.uniform(-5, 5)*2.5
    data.joint('j').qvel[1] = v[1]  # random.uniform(-5, 5)*2.5
    data.joint('j').qvel[2] = v[2]  # random.uniform(-5, 5)*2.5

    # refresh mujoco with new data
    mujoco.mj_forward(model, data)

    # get/calculate the camera vectors
    cam_pos = scene.camera[0].pos
    forward = scene.camera[0].forward
    up = scene.camera[0].up
    right = np.cross(up, forward)

    # calculate translational part
    cx = -np.dot(right, cam_pos)
    cy = -np.dot(up, cam_pos)
    cz = -np.dot(forward, cam_pos)

    # put together the extrinsic matrix
    ex_mat = np.matrix(
        [[right[0], right[1], right[2], cx], [up[0], up[1], up[2], cy], [forward[0], forward[1], forward[2], cz],
         [0, 0, 0, 1]])
    # print(ex_mat)

    # calculate focus length f
    fovy = math.radians(model.vis.global_.fovy)
    f = (0.5 * max_height) / math.tan(fovy / 2)

    # construct intrinisc matrix
    in_mat = np.matrix([[-f, 0, max_width / 2, 0], [0, f, max_height / 2, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
    # print(in_mat)

    if not os.path.isdir(output):
        os.makedirs(output)
    else:
        shutil.rmtree(output)
        os.makedirs(output)

    # simulation loop
    i = 0

    info_path = output + '/info.txt'
    with open(info_path, 'w') as info:
        info.write('E')
        info.write('\n')
        info.write(
            str(ex_mat[0, 0]) + ',' + str(ex_mat[0, 1]) + ',' + str(ex_mat[0, 2]) + ',' + str(ex_mat[0, 3]) + ',')
        info.write('\n')
        info.write(
            str(ex_mat[1, 0]) + ',' + str(ex_mat[1, 1]) + ',' + str(ex_mat[1, 2]) + ',' + str(ex_mat[1, 3]) + ',')
        info.write('\n')
        info.write(
            str(ex_mat[2, 0]) + ',' + str(ex_mat[2, 1]) + ',' + str(ex_mat[2, 2]) + ',' + str(ex_mat[2, 3]) + ',')
        info.write('\n')
        info.write(
            str(ex_mat[3, 0]) + ',' + str(ex_mat[3, 1]) + ',' + str(ex_mat[3, 2]) + ',' + str(ex_mat[3, 3]) + ',')
        info.write('\n')
        info.write('I ')
        info.write('\n')
        info.write(
            str(in_mat[0, 0]) + ',' + str(in_mat[0, 1]) + ',' + str(in_mat[0, 2]) + ',' + str(in_mat[0, 3]) + ',')
        info.write('\n')
        info.write(
            str(in_mat[1, 0]) + ',' + str(in_mat[1, 1]) + ',' + str(in_mat[1, 2]) + ',' + str(in_mat[1, 3]) + ',')
        info.write('\n')
        info.write(
            str(in_mat[2, 0]) + ',' + str(in_mat[2, 1]) + ',' + str(in_mat[2, 2]) + ',' + str(in_mat[2, 3]) + ',')
        info.write('\n')
        info.write(
            str(in_mat[3, 0]) + ',' + str(in_mat[3, 1]) + ',' + str(in_mat[3, 2]) + ',' + str(in_mat[3, 3]) + ',')

    label_path = output + '/label.csv'
    with open(label_path, 'w', newline='') as csvfile:
        fieldnames = ['t', 'camera x', 'camera y', 'camera z', 'ball x', 'ball y', 'ball z', 'ball vx', 'ball vy',
                      'ball vz', 'ball camx', 'ball camy', 'ball camz', 'ball imgx', 'ball imgy', 'ball visible']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        pos_list = []
        iteration = 0
        time_step = model.opt.timestep
        while data.time < record_time:
            if (abs(data.time * fps - iteration) < time_step * fps / 2):
                iteration += 1

                # convert ball position to the different coordinate systems: world -> camera -> image
                mujoco.mjv_updateScene(model, data, options, mujoco.MjvPerturb(), cam, mujoco.mjtCatBit.mjCAT_ALL,
                                       scene);
                ball_pos = np.array([data.geom('g').xpos[0], data.geom('g').xpos[1], data.geom('g').xpos[2], 1])
                pos_cam = np.dot(ex_mat, ball_pos)
                pos_image = np.dot(in_mat, np.transpose(pos_cam))
                pos_image = pos_image / pos_image[2]
                pos_image = np.transpose(pos_image)
                pos_list.append(pos_image)

                # print position
                # print("Position at time: " + str(data.time))
                # print(ball_pos)
                # print(pos_cam)
                # print(pos_image)

                # ball is outside of the frame -> redo the simulation
                if pos_image[0, 1] > max_height or pos_image[0, 1] < 0 or pos_image[0, 0] > max_width or pos_image[
                    0, 0] < 0:
                    csvfile.close()
                    sys.exit(100)
                #if the ball touches the floor -> redo the simulation
                if bounce == False and ball_pos[2] < 0.22:
                    csvfile.close()
                    sys.exit(100)

                ball_velocity = [data.body('freeball').cvel[-3], data.body('freeball').cvel[-2],
                                 data.body('freeball').cvel[-1]]
                writer.writerow({'t': str(data.time), 'camera x': str(cam_pos[0]), 'camera y': str(cam_pos[1]),
                                 'camera z': str(cam_pos[2]), 'ball x': str(ball_pos[0]), 'ball y': str(ball_pos[1]),
                                 'ball z': str(ball_pos[2]), 'ball vx': str(ball_velocity[0]),
                                 'ball vy': str(ball_velocity[1]), 'ball vz': str(ball_velocity[2]),
                                 'ball camx': str(pos_cam[0, 0]), 'ball camy': str(pos_cam[0, 1]),
                                 'ball camz': str(pos_cam[0, 2]), 'ball imgx': str(pos_image[0, 0]),
                                 'ball imgy': str(pos_image[0, 1]), 'ball visible': str(True)})

                # read pixels / construct image data
                mujoco.mjr_render(viewport, scene, ctx);
                upside_down_image = np.empty((max_height, max_width, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(upside_down_image, None, viewport, ctx)

                for p in pos_list:
                    if mark_pos and round(p[0, 1]) < (max_height - 1) and round(p[0, 0]) < (max_width - 1) and round(
                            p[0, 1]) > 0 and round(p[0, 0]) > 0:
                        upside_down_image[round(p[0, 1]), round(p[0, 0])] = 255

                # write frames as png files
                img = PIL.Image.fromarray(np.flipud(upside_down_image))
                path = output + '/image-' + str(i) + ".png"
                img.save(path)
                i = i + 1
            mujoco.mj_step(model, data)
    # save XML
    with open("environment.xml", "w") as myfile:
        myfile.write(XML)
    csvfile.close()


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str,
                        help='path to store simulated data')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed')
    parser.add_argument('--camera_mode', type=int, default=0,
                        help='camera angle mode')
    parser.add_argument('--xml_num', type=int, default=0,
                        help='xml number')

    args = parser.parse_args()
    assert args.camera_mode in list(range(len(camera_angles)))
    #subprocess.run('xvfb-run -a -s "-screen 0 1400x900x24" bash', shell=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    output = args.path
    XML = XMLs[args.xml_num]
    simulation(output, camera_mode=args.camera_mode, XML=XML)


if __name__ == '__main__':
    main()