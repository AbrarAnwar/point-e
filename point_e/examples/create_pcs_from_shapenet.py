from PIL import Image
import torch
from tqdm.auto import tqdm

import numpy as np
import open3d as o3d

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

OUTPUT="/data/abrar/3dgrounding/snare/data/pointe_pcs_hiddens"
RGB_PATH="/data/abrar/3dgrounding/snare/data/screenshots"


import imageio
import numpy as np
import sys, os, argparse, time
import open3d as o3d
import pickle
import json
import glob
import numpy as np
import multiprocessing as mp

def render_point_cloud(args, obj, sampler):
    batch_size = args.batch_size
    # load image
    files = glob.glob(os.path.join(RGB_PATH, obj, obj + '*.png'))
    # print(file)
    # base_path = file[:file.index('_r_')]
    # radius = file[file.index('_r_'):file.index('_r_')+7]

    if os.path.exists(os.path.join(OUTPUT, obj)):
        return
        # print()
    else:
        os.mkdir(os.path.join(OUTPUT, obj))

    batched = []
    get_int = lambda filename: int(filename.partition('.')[0].split('-')[-1])
    files.sort(key=get_int)

    file_offset = 6
    files = files[file_offset:]

    for i in range(0, len(files), batch_size):
        batched.append(files[i:i+batch_size])


    for num_batch, batched_files in enumerate(batched):
        
        # if os.path.exists(os.path.join(OUTPUT, obj, obj + '-' + num_file + '_pc.pcd')):
        #     continue


        # Load an image to condition on.
        imgs = []
        for file in batched_files:
            imgs.append(Image.open(file))

        # Produce a sample from the model.
        samples = None
        for x in tqdm(sampler.sample_batch_progressive(batch_size=len(imgs), model_kwargs=dict(images=imgs))):
            samples, hiddens = x
            
        pcs = sampler.output_to_point_clouds(samples)

        for pc_num, pc in enumerate(pcs):
            img_idx = batch_size*num_batch + pc_num # this should be right. double check!
            # import pdb; pdb.set_trace()

            coords = pc.coords
            R = pc.channels['R']
            G = pc.channels['G']
            B = pc.channels['B']
            rgb = np.stack([R,G,B])

            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coords))
            cloud.colors = o3d.utility.Vector3dVector(rgb.T)

            hidden_embed = hiddens[pc_num]
            with open(os.path.join(OUTPUT, obj, obj + '-' + str(img_idx) + '_hidden.npy'), 'wb') as f:
                np.save(f, hidden_embed)

            success = o3d.io.write_point_cloud(os.path.join(OUTPUT, obj, obj + '-' + str(img_idx) + '_pc.pcd'), cloud)
            if not success:
                print('Failed to write file to: ', os.path.join(OUTPUT, obj, obj +  '-' + str(img_idx) + '_pc.pcd'))


    return obj

def log_result(result):
    g_completed_jobs.append(result)
    elapsed_time = time.time() - g_starting_time

    if len(g_completed_jobs) % 1 == 0:
        msg = "%05d/%05d %s finished! " % (len(g_completed_jobs), g_num_total_jobs, result)
        msg = msg + 'Elapsed time: ' + \
                time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + '. '
        print(msg)

def main(args):
    train_train_files = ["/data/abrar/3dgrounding/snare/amt/folds_adversarial/train.json"]
    train_val_files = ["/data/abrar/3dgrounding/snare/amt/folds_adversarial/val.json"]
    test_test_files = ["/data/abrar/3dgrounding/snare/amt/folds_adversarial/test.json"]
    files = train_train_files + train_val_files + test_test_files
    # load amt data
    data = set()
    for file in files:
        loaded = json.load(open(file, 'r'))

        for instance in loaded:
            for object in instance['objects']:
                data.add(object)


    global g_completed_jobs
    global g_num_total_jobs
    global g_starting_time
    
    # data = ['ff64e45bc8ef04971e72001f9e1cdc7']
    data = list(data)
    data.sort(reverse=args.reverse)

    g_num_total_jobs = len(data)
    g_completed_jobs = []
    g_starting_time = time.time()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('creating base model...')
    base_name = 'base300M' # use base300M or base1B for better results
    # base_name = 'base1B' # use base300M or base1B for better results

    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    print('downloading base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, device))

    print('downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 3.0],
    )


    
    print('starting')

    if args.num_proc > 1:
        pool = mp.Pool(processes=args.num_proc) 
        print('Total jobs: %d, CPU num: %d' % (g_num_total_jobs, args.num_proc))
        for f in list(data):
            pool.apply_async(func=render_point_cloud, args=(args,f, sampler), callback=log_result)
        pool.close()
        pool.join()
    else:
        for f in list(data):
            render_point_cloud(args,f, sampler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--reverse", type=bool, default=False)

    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument('--gpu', nargs="+", type=str, default=[0],
                        help='GPUs to use. No CPU available since it is too slow. Must be an int') 
    args = parser.parse_args()

    main(args)