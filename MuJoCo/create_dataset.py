import os
import sys
import argparse
import time
from tqdm import tqdm
from multiprocessing import Pool

from paths import data_path


dataset_size = 100
num_camera_modes = 9
num_xmls = 3

#First run: xvfb-run -a -s "-screen 0 1400x900x24" bash
for script_name in ['MuJoCo/ball-falling.py', 'MuJoCo/ball-carousel.py', 'MuJoCo/ball-parcour.py']:
    for mode in ['train', 'val', 'test']:
        if 'carousel' in script_name or 'falling' in script_name:
            def wrapper(i):
                seed = 42 + i + 1
                if mode == 'test': seed += int(1e9)
                number_tries = 1
                exit_code = 100
                seed_var = 0
                while exit_code != 0:
                    data_name = 'ball_bouncing' if 'falling' in script_name else 'ball_carousel'
                    arguments = ('--path ' + os.path.join(data_path, f'{data_name}', mode,
                                                          str(i)), '--seed ' + str(seed + seed_var))
                    args = ''
                    for a in arguments: args += a + ' '
                    cmd = ''.join(('python ', script_name, ' ', args))
                    exit_code = os.system(cmd)
                    time.sleep(0.2)
                    if exit_code != 0:
                        number_tries += 1
                        seed_var = dataset_size * number_tries
                if number_tries > 1: print(f'tried {number_tries} times')
                time.sleep(0.5)

            with Pool(10) as p:
                __ = list(tqdm(p.imap(wrapper, range(dataset_size)), total=dataset_size))
        elif 'parcour' in script_name:
            def wrapper(stuff):
                i, camera_mode, xml_num = stuff
                seed = 42 + i + 1 + int(1e5) * camera_mode + int(1e8) * xml_num
                if mode == 'test': seed += int(1e9)
                elif mode == 'val': seed += 2 * int(1e9)
                number_tries = 1
                exit_code = 100
                seed_var = 0
                while exit_code != 0:
                    arguments = ('--path ' + os.path.join(data_path, 'ball_parcour_new', mode,
                                                          f'xml{xml_num}', f'camera{camera_mode}', str(i)),
                                 '--seed ' + str(seed + seed_var),
                                 f'--camera_mode {camera_mode}',
                                 f'--xml_num {xml_num}'
                                 )
                    args = ''
                    for a in arguments: args += a + ' '
                    cmd = ''.join(('python ', script_name, ' ', args))
                    exit_code = os.system(cmd)
                    time.sleep(0.2)
                    if exit_code != 0:
                        number_tries += 1
                        seed_var = dataset_size * number_tries
                if number_tries > 1: print(f'tried {number_tries} times')
                time.sleep(0.5)

            for xml_num in range(num_xmls):
                for cam_mode in range(num_camera_modes):
                    with Pool(10) as p:
                        __ = list(tqdm(p.imap(wrapper, iter([(i, cam_mode, xml_num) for i in range(dataset_size)])), total=dataset_size))
        else:
            raise ValueError('Wrong script name')


