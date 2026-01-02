import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import yaml
import numpy.random as random
from torch.utils.data.dataloader import DataLoader
from datasets.megadepth import MegaDepth
from datasets.scannet import Scannet
from datasets.yfcc import Yfcc
from models.first_layer import SuperGlue_new
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from utils.metrics import compute_pose_error, aggregate_metrics
import torchvision
import time


@torch.no_grad()
def evaluate_megadepth(model, data_path, pairs_path, scale_factor, threshold):
    dataset = MegaDepth(data_path, pairs_path, is_train=False, aug_resolution=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    error_R_list, error_t_list, result_list, result_mask_list = [], [], [], []
    time_list = []
    macs_list = []
    params = None
    total_num = sum(p.numel() for p in model.parameters())
    print("total_param_num:", total_num)
    for i, data in tqdm(enumerate(loader)):
        # if i > 100: break
        data['image0'] = data['image0'].cuda()
        data['image1'] = data['image1'].cuda()
        img = torch.cat([data['image0'], data['image1']], 0)
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = normalize(img.permute(0, 3, 1, 2).float().contiguous()/ 255.0)
        # for i in range(10):
        #     results, _ = model(img, True)
        torch.cuda.synchronize()
        start = time.time()
        results, _ = model(img, True)
        torch.cuda.synchronize()
        end = time.time()
        # macs_list.append(macs)
        time_list.append(end - start)
        matches_l = results[-1]['mkpts0_f'][results[-1]['if_matching2']].cpu().numpy()
        matches_r = results[-1]['mkpts1_f'][results[-1]['if_matching2']].cpu().numpy()
        # matches_l = results[-1]['mkpts0_f'].cpu().numpy()[:, :2]
        # matches_r = results[-1]["mkpts1_f"].cpu().numpy()[:, :2]
        error_R, error_t, result, result_mask = compute_pose_error(matches_l,
                                              matches_r,
                                              data['K0'][0].numpy(), data['K1'][0].numpy(),
                                              data['T0'][0].numpy(), data['T1'][0].numpy(),
                                              scale_factor, 0.25)
        error_R_list.append(error_R)
        error_t_list.append(error_t)
        result_list.append(result.mean())
        result_mask_list.append(result_mask.mean())
    metric = aggregate_metrics(error_R_list, error_t_list)
    print(np.mean(np.asarray(result_list)))
    print(np.mean(np.asarray(result_mask_list)))
    print('-'*5 + 'Evaluation on MegaDepth' + '-'*5)
    # avg_macs = np.array(macs_list).mean()
    # avg_macs, params = clever_format([avg_macs, params], "%.3f")
    # print("MACs:", avg_macs, "Params:", params )
    print("avg time:", np.array(time_list).mean())
    for key, value in metric.items():
        print(f'{key}: {value}')


@torch.no_grad()
def evaluate_yfcc(model, data_path, pairs_path, scale_factor, threshold):
    dataset = Yfcc(data_path, pairs_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    error_R_list, error_t_list = [], []
    time_list = []
    for i, data in tqdm(enumerate(loader)):
        # if i > 100: break
        data['image0'] = data['image0'].cuda()
        data['image1'] = data['image1'].cuda()
        img = torch.cat([data['image0'], data['image1']], 0)
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = normalize(img.permute(0, 3, 1, 2).float().contiguous()/ 255.0)
        # for i in range(10):
        # results, _ = model(img, True)
        torch.cuda.synchronize()
        start = time.time()
        results, _ = model(img, True)
        torch.cuda.synchronize()
        end = time.time()
        time_list.append(end - start)
        error_R, error_t, result, result_mask = compute_pose_error(results[-1]['mkpts0_f'][results[-1]['if_matching2']].cpu().numpy(),
                                              results[-1]['mkpts1_f'][results[-1]['if_matching2']].cpu().numpy(),
                                              data['K0'][0].numpy(), data['K1'][0].numpy(),
                                              data['T0'][0].numpy(), data['T1'][0].numpy(),
                                              scale_factor, threshold)
        error_R_list.append(error_R)
        error_t_list.append(error_t)
    metric = aggregate_metrics(error_R_list, error_t_list)
    print('-'*5 + 'Evaluation on YFCC' + '-'*5)
    print("avg time:", np.array(time_list).mean())
    for key, value in metric.items():
        print(f'{key}: {value}')



@torch.no_grad()
def evaluate_scannet(model, data_path, pairs_path, scale_factor, threshold):
    dataset = Scannet(data_path, pairs_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    error_R_list, error_t_list = [], []
    time_list = []
    for i, data in tqdm(enumerate(loader)):
        if i > 100: break
        data['image0'] = data['image0'].cuda()
        data['image1'] = data['image1'].cuda()
        img = torch.cat([data['image0'], data['image1']], 0)
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = normalize(img.permute(0, 3, 1, 2).float().contiguous()/ 255.0)
        results, _ = model(img, True)
        torch.cuda.synchronize()
        start = time.time()
        results, _ = model(img, True)
        torch.cuda.synchronize()
        end = time.time()
        time_list.append(end - start)
        error_R, error_t, result, result_mask = compute_pose_error(results[-1]['mkpts0_f'][results[-1]['if_matching2']].cpu().numpy(),
                                              results[-1]['mkpts1_f'][results[-1]['if_matching2']].cpu().numpy(),
                                              data['K0'][0].numpy(), data['K1'][0].numpy(),
                                              data['T0'][0].numpy(), data['T1'][0].numpy(),
                                              scale_factor, threshold)
        error_R_list.append(error_R)
        error_t_list.append(error_t)
    metric = aggregate_metrics(error_R_list, error_t_list)
    print('-'*5 + 'Evaluation on ScanNet' + '-'*5)
    print("avg time:", np.array(time_list).mean())
    for key, value in metric.items():
        print(f'{key}: {value}')

def resize_to_divisible(image, divisible=32):
    """
    Resize the image such that both width and height are divisible by the given number.
    If the image's dimensions are already divisible, pad with zeros.
    """
    height, width, _ = image.shape

    # Calculate target dimensions that are the smallest multiple of divisible
    target_height = ((height + divisible - 1) // divisible) * divisible
    target_width = ((width + divisible - 1) // divisible) * divisible

    # Create an empty array of zeros with target dimensions
    target_image = np.zeros((target_height, target_width, 3), dtype=image.dtype)

    # Copy the input image onto the center of the target image
    y_offset = (target_height - height) // 2
    x_offset = (target_width - width) // 2
    target_image[y_offset:y_offset+height, x_offset:x_offset+width] = image

    return target_image



if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--scale_factor', type=float, default=1.0)
    param = parser.parse_args()
    param.cur_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if param.config is not None:
        with open(param.config, 'r') as f:
            yaml_dict = yaml.safe_load(f)
            for k, v in yaml_dict.items():
                param.__dict__[k] = v
    # initialize random seed
    torch.manual_seed(param.seed)
    np.random.seed(param.seed)
    random.seed(param.seed)
    model = SuperGlue_new()
    model_dict = torch.load(param.checkpoint, map_location= "cuda:0")
    new_model_dict = model.state_dict()
    for k, v in model_dict.items():
        name = k[7:]
        new_model_dict[name] = v
    model.load_state_dict(new_model_dict)
    model = model.cuda().eval()
    dataset_name = param.dataset
    print("Evaluate Ours")
    if dataset_name == 'MegaDepth':
        evaluate_megadepth(model, param.data_path, param.pairs_path, param.scale_factor, param.threshold)
    elif dataset_name == 'YFCC':
        evaluate_yfcc(model, param.data_path, param.pairs_path, param.scale_factor, param.threshold)
    elif dataset_name == 'ScanNet':
        evaluate_scannet(model, param.data_path, param.pairs_path, param.scale_factor, param.threshold)
