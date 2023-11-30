
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


# try running this command to make rendering work by creating an in-memory frame buffer
#   xvfb-run -a -s "-screen 0 1400x900x24" bash
def show_environment(xml_number, cam_num):
    import mujoco
    import numpy as np
    import PIL.Image

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
    camera_angle = camera_angles[cam_num]
    XML = XMLs[xml_number]
    gl = mujoco.GLContext(224, 224)
    gl.make_current()
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scene = mujoco.MjvScene(model, maxgeom=100)
    cam = mujoco.MjvCamera()
    options = mujoco.MjvOption()
    cam.lookat[0] = 0
    cam.lookat[1] = 0
    cam.lookat[2] = 0
    cam.azimuth = camera_angle[0]
    cam.elevation = camera_angle[1]
    cam.distance = camera_angle[2]
    mujoco.mjv_updateScene(model, data, options, mujoco.MjvPerturb(), cam, mujoco.mjtCatBit.mjCAT_ALL, scene);
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, ctx)
    viewport = mujoco.MjrRect(0, 0, 224, 224)
    mujoco.mj_forward(model, data)
    mujoco.mjv_updateScene(model, data, options, mujoco.MjvPerturb(), cam, mujoco.mjtCatBit.mjCAT_ALL,
                           scene);
    mujoco.mjr_render(viewport, scene, ctx);
    upside_down_image = np.empty((224, 224, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(upside_down_image, None, viewport, ctx)

    # write frames as png files
    img = PIL.Image.fromarray(np.flipud(upside_down_image))
    path =  f"tmp{cam_num}.png"
    img.save(path)


if __name__ == "__main__":
    for cam_num in range(9):
        show_environment(2, cam_num)