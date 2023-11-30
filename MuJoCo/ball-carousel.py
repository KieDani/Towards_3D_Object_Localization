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


# Troubleshooting notes(WSL):
# try running this command to make rendering work by creating an in-memory frame buffer
#   xvfb-run -a -s "-screen 0 1400x900x24" bash
# if problems persist, the following command might help but should not be necessary for MuJoCo 2.1.4 and higher (GLEW was replaced with GLAD)
#   export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

# General settings:
fps = 15  # frames per second
record_time = 8  # in seconds
mark_pos = False  # will mark the path of the small ball as it travels
max_width = 224  # screen width
max_height = 224  # screen height
dataset_size = 1000

# XML Model
XML = r"""
<mujoco>
  <option gravity="0 0 0">
  </option>
  <asset>
    <material name="floor" reflectance=".1"/>
  </asset>
  <worldbody>
        <light diffuse=".5 .5 .5" pos="-0.5 -0.5 5" dir="0.1 0.1 -1"/>
        <geom type="plane" size="500 500 0.1" rgba="0.6 0.6 0.6 1" pos="0 0 -10" material="floor"/>
    <body pos="0 0 0" name="centerball">
      <joint name="c" type="ball"/>
      <geom name="g1" type="sphere" size=".3" rgba="0 0 .9 1"/>
      <site type="sphere" size="0.02" rgba="1 0 0 1" pos="0.0 0 0" name="b1"/>
    </body>
    <body pos="1 0 0" name="freeball">
      <joint type="free" name="j"/>
      <geom type="sphere" size=".2" mass="1.0" rgba="0 .9 0 1" name="g2"/>
      <site type="sphere" size="0.02" rgba="1 0 0 1" pos="-0.0 0 0" name="b2"/>
    </body>
  </worldbody>
  <tendon> 
    <spatial name="spring" width="0.02" rgba=".95 .3 .3 1" springlength="0.0" stiffness="3" frictionloss="0" damping="0" limited="false" range="0 2">
      <site site="b1"/>
      <site site="b2"/>
    </spatial>
  </tendon>
</mujoco>
"""
#in case of any weirdness switch ball joint to hinge
#damping="100"
#springlength
#limited="true" range="0.15 2"
# give out helpful information when prompted
if len(sys.argv) >= 2 and (sys.argv[1] == 'help' or sys.argv[1] == '-h' or sys.argv[1] == '-help'):
	print('This script will simulate 2 balls, connected by a spring, with one of them circling the other for 6 seconds, as well as label data. Refer to the contents of the script for more details.')
	print('Usage: ball-carousel.py <input .csv> <output folder> <optional render origin flag>')
	print('Input is optional, if not provided, ball position and velocity will be randomized. Inputting random allows for random data while also being able to specify an output folder.')
	print('Output folder defaults to creating a folder named \'output\'. Output files are recorded frames in .png format and label information in .csv format.')
	print('Set render origin flag to 1 if the global coordinate system origin should be rendered with its axis.')
	print('When encountering errors related to rendering, consult \'Troubleshooting notes\' inside the script.')
	sys.exit()

def simulation(output):
    # init mujoco + GLcontext
    #print("Mujoco Version " + mujoco.mj_versionString())

    gl = mujoco.GLContext(max_width, max_height)
    gl.make_current()

    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    scene = mujoco.MjvScene(model, maxgeom=100)
    cam = mujoco.MjvCamera()
    options = mujoco.MjvOption()

    options.frame = mujoco.mjtFrame.mjFRAME_WORLD

    # camera settings
    # defines camera position via a spherical coordinate system, using lookat as the origin:
    # default camera settings
    cam.lookat[0] = 0
    cam.lookat[1] = 0
    cam.lookat[2] = 0
    cam.azimuth = 40
    cam.elevation = -60
    cam.distance = 7

    # init graphics related stuff
    mujoco.mjv_updateScene(model, data, options, mujoco.MjvPerturb(), cam, mujoco.mjtCatBit.mjCAT_ALL, scene);
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, ctx)
    viewport = mujoco.MjrRect(0, 0, max_width, max_height)

    # randomize the size of the ball between 0.1 and 0.5
    #ball_size = random.randrange(10, 50) / 100
    #model.geom_size[1] = ball_size
    #print("Ball size: " + str(ball_size))

    # init ball with data
    # if no data was read, generate random data with some simple constraints
    # to make sure the ball does actually cross through the image
    sgn = np.where(np.random.randint(0, 1, size=3)  < 0.5, -1, 1)
    r = data.joint('c').qpos[:3] + sgn * np.random.uniform(0.5, 1.5, size=3)
    data.joint('j').qpos[0] = r[0] #random.uniform(-2, 2)
    data.joint('j').qpos[1] = r[1] #random.uniform(-2, 2)
    data.joint('j').qpos[2] = r[2] #random.uniform(-2, 2)

    #set initial velocity v orthogonal to r (but not exactly othogonal)
    r = r - data.joint('c').qpos[:3]
    e_r = r / np.linalg.norm(r)
    sgn = np.where(np.random.randint(0, 1, size=3)  < 0.5, -1, 1)
    v = sgn * np.random.uniform(2.0, 20.0, size=3)
    e_v = v / np.linalg.norm(v)
    e_v -= np.dot(e_r, e_v) * e_r
    e_v = e_v / np.linalg.norm(e_v)
    #for circle -> v = sqrt(k/m) * r
    sgn = np.where(np.random.randint(0, 1, size=3) < 0.5, -1, 1)
    v = sgn * math.sqrt(3/1) * np.linalg.norm(r) *  e_v
    v += np.random.uniform(-0.75 * np.linalg.norm(v), 0.75 * np.linalg.norm(v), size=3)
    data.joint('j').qvel[0] = v[0] #random.uniform(-5, 5)*2.5
    data.joint('j').qvel[1] = v[1] #random.uniform(-5, 5)*2.5
    data.joint('j').qvel[2] = v[2] #random.uniform(-5, 5)*2.5

    # if data.joint('j').qpos[0] < 0.3 and data.joint('j').qpos[0] > -0.3:
    #     data.joint('j').qpos[0] = 10 * data.joint('j').qpos[0]
    # if data.joint('j').qpos[1] < 0.3 and data.joint('j').qpos[1] > -0.3:
    #     data.joint('j').qpos[1] = 10 * data.joint('j').qpos[1]
    # if data.joint('j').qpos[2] < 0.3 and data.joint('j').qpos[2] > -0.3:
    #     data.joint('j').qpos[2] = 10 * data.joint('j').qpos[2]
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
    #print(ex_mat)

    # calculate focus length f
    fovy = math.radians(model.vis.global_.fovy)
    f = (0.5 * max_height) / math.tan(fovy / 2)

    # construct intrinisc matrix
    in_mat = np.matrix([[-f, 0, max_width / 2, 0], [0, f, max_height / 2, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
    #print(in_mat)

    if not os.path.isdir(output):
        os.makedirs(output)
    else:
        shutil.rmtree(output)
        os.makedirs(output)

    # simulation loop
    i = 0
    frametime = 0

    info_path = output + '/info.txt'
    with open(info_path, 'w') as info:
        info.write('E')
        info.write('\n')
        info.write(str(ex_mat[0, 0]) + ',' + str(ex_mat[0, 1]) + ',' + str(ex_mat[0, 2]) + ',' + str(ex_mat[0, 3]) + ',')
        info.write('\n')
        info.write(str(ex_mat[1, 0]) + ',' + str(ex_mat[1, 1]) + ',' + str(ex_mat[1, 2]) + ',' + str(ex_mat[1, 3]) + ',')
        info.write('\n')
        info.write(str(ex_mat[2, 0]) + ',' + str(ex_mat[2, 1]) + ',' + str(ex_mat[2, 2]) + ',' + str(ex_mat[2, 3]) + ',')
        info.write('\n')
        info.write(str(ex_mat[3, 0]) + ',' + str(ex_mat[3, 1]) + ',' + str(ex_mat[3, 2]) + ',' + str(ex_mat[3, 3]) + ',')
        info.write('\n')
        info.write('I ')
        info.write('\n')
        info.write(str(in_mat[0, 0]) + ',' + str(in_mat[0, 1]) + ',' + str(in_mat[0, 2]) + ',' + str(in_mat[0, 3]) + ',')
        info.write('\n')
        info.write(str(in_mat[1, 0]) + ',' + str(in_mat[1, 1]) + ',' + str(in_mat[1, 2]) + ',' + str(in_mat[1, 3]) + ',')
        info.write('\n')
        info.write(str(in_mat[2, 0]) + ',' + str(in_mat[2, 1]) + ',' + str(in_mat[2, 2]) + ',' + str(in_mat[2, 3]) + ',')
        info.write('\n')
        info.write(str(in_mat[3, 0]) + ',' + str(in_mat[3, 1]) + ',' + str(in_mat[3, 2]) + ',' + str(in_mat[3, 3]) + ',')

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
            if (abs(data.time * fps - iteration) < time_step * fps / 2 ):
                iteration += 1

                # convert ball position to the different coordinate systems: world -> camera -> image
                mujoco.mjv_updateScene(model, data, options, mujoco.MjvPerturb(), cam, mujoco.mjtCatBit.mjCAT_ALL, scene);
                ball_pos = np.array([data.geom('g2').xpos[0], data.geom('g2').xpos[1], data.geom('g2').xpos[2], 1])
                center_ball_pos = np.array([data.geom('g1').xpos[0], data.geom('g1').xpos[1], data.geom('g1').xpos[2], 1])
                pos_cam = np.dot(ex_mat, ball_pos)
                pos_image = np.dot(in_mat, np.transpose(pos_cam))
                pos_image = pos_image / pos_image[2]
                pos_image = np.transpose(pos_image)
                pos_list.append(pos_image)

                pos_centerball = np.array([0, 0, 0, 1])
                pos_centerball = np.dot(ex_mat, pos_centerball)

                # print position
                #print("Position at time: " + str(data.time))
                #print(ball_pos)
                #print(pos_cam)
                #print(pos_image)

                visible = True
                #ball is behind center ball
                if (pos_centerball[0, 0]-0.15 < pos_cam[0, 0] < pos_centerball[0, 0]+0.15) and (pos_centerball[0, 1]-0.15 < pos_cam[0, 1] < pos_centerball[0, 1]+0.15) and pos_cam[0, 2] > pos_centerball[0, 2]:
                    visible = False
                #ball is outside of the frame -> redo the simulation
                if pos_image[0, 1] > max_height or pos_image[0, 1] < 0 or pos_image[0, 0] > max_width or pos_image[0, 0] < 0:
                    csvfile.close()
                    sys.exit(100)
                #collision of the two balls
                if np.linalg.norm(center_ball_pos[:] - ball_pos[:]) < 0.85:
                    csvfile.close()
                    sys.exit(100)

                ball_velocity = [data.body('freeball').cvel[-3], data.body('freeball').cvel[-2], data.body('freeball').cvel[-1]]
                writer.writerow({'t': str(data.time), 'camera x': str(cam_pos[0]), 'camera y': str(cam_pos[1]),
                                 'camera z': str(cam_pos[2]), 'ball x': str(ball_pos[0]), 'ball y': str(ball_pos[1]),
                                 'ball z': str(ball_pos[2]), 'ball vx': str(ball_velocity[0]),
                                 'ball vy': str(ball_velocity[1]), 'ball vz': str(ball_velocity[2]),
                                 'ball camx': str(pos_cam[0, 0]), 'ball camy': str(pos_cam[0, 1]),
                                 'ball camz': str(pos_cam[0, 2]), 'ball imgx': str(pos_image[0, 0]),
                                 'ball imgy': str(pos_image[0, 1]), 'ball visible' : str(visible)})

                # read pixels / construct image data
                mujoco.mjr_render(viewport, scene, ctx);
                upside_down_image = np.empty((max_height, max_width, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(upside_down_image, None, viewport, ctx)

                for p in pos_list:
                    if mark_pos and round(p[0, 1]) < (max_height-1) and round(p[0, 0]) < (max_width-1) and round(p[0, 1]) > 0 and round(p[0, 0]) > 0:
                        upside_down_image[round(p[0, 1]), round(p[0, 0])] = 255

                # write frames as png files
                img = PIL.Image.fromarray(np.flipud(upside_down_image))
                path = output + '/image-' + str(i) + ".png"
                img.save(path)
                i = i + 1
            mujoco.mj_step(model, data)
    csvfile.close()


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str,
                        help='path to store simulated data')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed')

    args = parser.parse_args()
    #subprocess.run('xvfb-run -a -s "-screen 0 1400x900x24" bash', shell=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    output = args.path
    simulation(output)


if __name__ == '__main__':
    main()