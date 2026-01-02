#sftp
import warnings
warnings.filterwarnings("ignore")
import torch
import os
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import yaml
import numpy.random as random
from torch.utils.data.dataloader import DataLoader
from dataset import MegaData3, ScanNet, data_prefetcher, img_show, img_show_activate_first, img_show_step2, draw_lines, refine_result_show, img_show_errors1, img_show_errors2
from models.first_layer import SuperGlue_new
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from utils.utils import project_function, split_patches, get_result, matches_show, Compute_new_img, Compute_accuracy, Get_cameras
from utils.coco_utils import Compute_resized_homography, Compute_resized_refine_dense, eval_epipolar_label,\
    Compute_resized_epipolar, Compute_dense_label_H, Compute_dense_refine_label_H, Compute_accuracy_indicator,\
    Extract_refine
# from my_third_party.superglue.models.superpoint import SuperPoint
# from my_third_party.superglue.models.superglue import SuperGlue
# from my_third_party.LoFTR.src.loftr import LoFTR, default_cfg
from copy import deepcopy
# from models.loftr_fine import LoFTR_fine
import torch.nn.functional as F
import sys
from datasets.scannet_extract import create_megadepth_label
# sys.path.append("../my_third_party/DenseMatching/")
# from my_third_party.DenseMatching.model_selection import select_model
# from my_third_party.DenseMatching.utils_flow.flow_and_mapping_operations import convert_flow_to_mapping
# from my_third_party.DenseMatching.validation.utils import matches_from_flow
# from my_third_party.DenseMatching.models.inference_utils import estimate_mask
from queue import Queue,LifoQueue,PriorityQueue
import time
# os.environ["OMP_NUM_THREADS"] = str(1)
import torchvision

def draw_grad(named_parameters):
    fig = plt.figure(figsize=(10, 8), dpi=100)
    plt.clf()
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            if not hasattr(p.grad, 'abs'):
                continue
            layers.append(n[:-7])
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.plot(max_grads, alpha=0.3, color="c")
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("max / average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    return fig


def process_regressor(model, data, params, type="epipolar", if_choose=False, sequence_num=0):
    (P, left, right, depth_left, depth_right, left_path, right_path) = data
    with torch.no_grad():
        label, reverse_label = create_megadepth_label(P, depth_left, depth_right)
    img = torch.cat([left, right], 0)
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(img.permute(0, 3, 1, 2).float().contiguous())
    # img_show(left[1].cpu().numpy(), right[1].cpu().numpy(), label[1].cpu().numpy(), sequence_num)
    # (results, results_second, gather_homography) = model(left, right)
    # torch.cuda.synchronize()
    # start = time.time()    
    (results, gather_homography) = model(img)
    # torch.cuda.synchronize()
    # end = time.time()
    # print(end-start)
    (loss, medium_information) = SuperGlue_new.loss_function(model, label,
        reverse_label, left, right, eval_scores=None, refine_mode=False, if_choose=if_choose, gather_homography=gather_homography, 
        results=results)
    # (loss, medium_information) = SuperGlue_new.loss_function(model, label,
    #     reverse_label, left, right, eval_scores=None, refine_mode=True, if_choose=if_choose, gather_homography=gather_homography, 
    #     results=results_second)
    # weight = [1.0, 0.0, 0.0]
    weight = [1e2, 1.5, 20.0, 5.0, 100.0, 10.0]
    loss_whole = weight[0] * loss['classify'] + weight[1] * loss['position'] + weight[2] * loss['nomatching'] + weight[3] * loss['seg_loss'] +\
        weight[4] * loss['position_refine'] + weight[5] * loss["confidence_refine"]
    # print(weight[0] * loss['classify'], weight[1] * loss['position'], weight[2] * loss['nomatching'], weight[3] * loss['seg_loss'], 
    #     weight[4] * loss['position_refine'], weight[5] * loss['classify_refine'])
    with torch.no_grad():
        accuracy1, accuracy2, accuracy3, position_error = Compute_accuracy_indicator(medium_information["average_error1"], 
            medium_information["label_use"], medium_information['classify_accurate'], medium_information["choice_rate"], if_second=False)
    medium_information["left_path"] = left_path
    medium_information["right_path"] = right_path
    return (loss_whole, accuracy1, accuracy2, accuracy3, position_error, medium_information)

def train(param):
    type = param.loss_type
    # writer = SummaryWriter(log_dir=os.path.join(param.log_dir, param.cur_time))
    # writer.add_text('parameters', param2str(param.__dict__))
    saver = param.saver
    model = param.model
    # model, train_dataset, val_loader, param, device, length, length2, writer, saver = args
    # dist.init_process_group(backend='nccl', init_method='env://', world_size=np.array(param.gpu_ids).size, rank=param.local_rank)
    torch.cuda.set_device(param.local_rank)
    model.cuda(param.local_rank)
    device = torch.cuda.current_device()
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if ("local_segment" not in n and p.requires_grad) and ("fine_preprocess" not in n and p.requires_grad) and\
            ("compress2" not in n and p.requires_grad) and ("kenc_local" not in n and p.requires_grad) and ("kenc_hypo" not in n and p.requires_grad)]},
        {
            "params": [p for n, p in model.named_parameters() if ("local_segment" in n and p.requires_grad) or ("compress2" in n and p.requires_grad)\
                or ("kenc_local" in n and p.requires_grad) or ("kenc_hypo" in n and p.requires_grad) or ("fine_preprocess" in n and p.requires_grad)],
            "lr": param.backbone_lr,
        }
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=param.lr,
                                 weight_decay=param.weight_decay)
    # lambda1 = lambda epoch: (epoch + 1)/3 if epoch <= 2 else 1.0 if epoch < param.milestones[0] else param.gamma
    # lambda1 = lambda epoch: param.gamma
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
    scheduler = MultiStepLR(optimizer, milestones=param.milestones, gamma=param.gamma)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[param.local_rank], find_unused_parameters=True)
    # scaler = GradScaler()
    seed = param.seed
    test_dataset = MegaData3(label_path=param.label_path, label_reverse_path=param.label_reverse_path, pairs_path=param.pairs_path, is_train=False,
                                data_path=param.origin_data_path)
    test_loader = DataLoader(test_dataset, batch_size=1,
                        num_workers=param.num_workers, drop_last=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(param.train_dataset,
                    num_replicas=np.array(param.gpu_ids).size, rank=param.local_rank, shuffle=True)
    train_loader = DataLoader(param.train_dataset, batch_size=param.batch_size,
                    num_workers=param.num_workers, sampler=train_sampler, pin_memory=True, drop_last=True)
    param.l1 = train_loader.__len__()
    # prefetch_loader = data_prefetcher(train_loader, device)
    for epoch in range(param.num_epoch):
        # model.eval()
        model.train()
        # import ipdb
        # ipdb.set_trace()
        # long running
        # do something other
        prefetch_loader = data_prefetcher(train_loader, device)
        data = Queue(maxsize=2)
        for i in range(1):
            data.put(prefetch_loader.next())
        if epoch >= 100:
            if_choose = True
        else:
            if_choose = False
        seed = seed + 100
        total_train_dist = {"whole_loss": torch.tensor(0.0, device=device),
                            "accuracy1": torch.tensor(0.0, device=device),
                            "accuracy2": torch.tensor(0.0, device=device),
                            "accuracy3": torch.tensor(0.0, device=device),
                            "position_loss": torch.tensor(0.0, device=device), }
        num = 0
        for i in tqdm(range(int(param.l1 * 1.0)), ncols=50):
            # with autocast():
            # loss, loss_matches, loss_scale, accuracy1, accuracy2 = process_regressor(model, data, param)
            # if data.qsize() <= 1:
            #     torch.cuda.synchronize()
            loss, accuracy1, accuracy2, accuracy3, position_loss, result = process_regressor(model, data.get(), param, type=type, if_choose=if_choose, sequence_num=i)
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            data.put(prefetch_loader.next())
            num += 1
            total_train_dist['whole_loss'] += loss.item()
            total_train_dist['accuracy1'] += accuracy1.item()
            total_train_dist['accuracy2'] += accuracy2.item()
            total_train_dist['accuracy3'] += accuracy3.item()
            total_train_dist['position_loss'] += position_loss.sum().item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
            optimizer.zero_grad()
            # endtime = datetime.now()
            # print((endtime - starttime).microseconds)
        print("epoch {} - whole_loss {}"
              .format(epoch,
                      total_train_dist['whole_loss'].item() / (num + 1e-9)))
        print("epoch {} - loss_ratio {}"
              .format(epoch,
                      total_train_dist['accuracy1'].item() / (num + 1e-9)))
        print("epoch {} - exist_accuracy {}"
              .format(epoch,
                      total_train_dist['accuracy2'].item() / (num + 1e-9)))
        print("epoch {} - none_accuracy {}"
              .format(epoch,
                      total_train_dist['accuracy3'].item() / (num + 1e-9)))
        print("epoch {} - position_loss {}"
              .format(epoch,
                      total_train_dist['position_loss'].item() / (num + 1e-9)))
        scheduler.step()
        if (saver is not None and param.local_rank == 0):
            saver.add_checkpoint(model, epoch)
            # test(model, test_loader, param, device, epoch)

def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    dist.barrier()
    for param in model.parameters():
        if param.grad != None:
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size

@torch.no_grad()
def test(model, loader, param, device, epoch=None, writer=None, type="epipolar"):
    total_test_dist = {"whole_loss": torch.tensor(0.0, device=device),
                       "accuracy1": torch.tensor(0.0, device=device),
                       "accuracy2": torch.tensor(0.0, device=device),
                       "accuracy3": torch.tensor(0.0, device=device),
                       "position_loss": torch.tensor(0.0, device=device), }
    model.eval()
    prefetch_loader = data_prefetcher(loader, device)
    data = prefetch_loader.next()
    length = loader.__len__()
    n = 0
    results = []
    for i in tqdm(range(length), ncols=50):
        loss, accuracy1, accuracy2, accuracy3, position_loss, result = process_regressor(model, data, param, type=type, sequence_num=n)
        results.append(result)
        total_test_dist['whole_loss'] += loss.item()
        total_test_dist['accuracy1'] += accuracy1.item()
        total_test_dist['accuracy2'] += accuracy2.item()
        total_test_dist['accuracy3'] += accuracy3.item()
        total_test_dist['position_loss'] += position_loss.sum().item()
        data = prefetch_loader.next()
    cameras = Get_cameras(param.pairs_path, param.origin_data_path, "test")
    # Compute_accuracy(dataset.images, results)
    Compute_accuracy(cameras, results, scale_factor=1.0, threshold=0.25)

    print("epoch {} - whole_loss {}"
            .format(epoch,
                    total_test_dist['whole_loss'].item() / (length + 1e-9)))
    print("epoch {} - loss_ratio {}"
            .format(epoch,
                    total_test_dist['accuracy1'].item() / (length + 1e-9)))
    print("epoch {} - exist_accuracy {}"
            .format(epoch,
                    total_test_dist['accuracy2'].item() / (length + 1e-9)))
    print("epoch {} - none_accuracy {}"
            .format(epoch,
                    total_test_dist['accuracy3'].item() / (length + 1e-9)))
    print("epoch {} - position_loss {}"
            .format(epoch,
                    total_test_dist['position_loss'].item() / (length + 1e-9)))
    # if writer is not None:
    #     writer.add_scalars('total',
    #                        {'whole_loss': total_test_dist[
    #                                           'whole_loss'].item() / length,
    #                         },
    #                        global_step=epoch)
    return total_test_dist['whole_loss'].item() / length


def param2str(param_dict):
    param_str = ''
    for k, v in param_dict.items():
        param_str += '{}: {} \t\n'.format(k, v)
    return param_str


class Saver:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.smallest_metric = 1000.0
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, )

    def add_checkpoint(self, model, epoch, metric = 0):
        # if metric > self.smallest_metric:
        #     return
        # else:
        #     self.smallest_metric = metric
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), os.path.join(self.save_dir, 'cp_{}.pth'.format(epoch)))
        else:
            torch.save(model.state_dict(), os.path.join(self.save_dir, 'cp_{}.pth'.format(epoch)))

        print("save for epoch {} with metric {}".format(epoch, metric))


    def add_checkpoint2(self, model, epoch, metric = 0):
        # if metric > self.smallest_metric:
        #     return
        # else:
        #     self.smallest_metric = metric
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), os.path.join(self.save_dir, 'cp_{}_refine.pth'.format(epoch)))
        else:
            torch.save(model.state_dict(), os.path.join(self.save_dir, 'cp_{}_refine.pth'.format(epoch)))

        print("save for epoch {} with metric {}".format(epoch, metric))


def init_distributed_mode(args):
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.local_rank, "env://"), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method="tcp://localhost:23451",
                                         world_size=np.array(param.gpu_ids).size, rank=args.local_rank)
    torch.distributed.barrier()
    setup_for_distributed(args.local_rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print

if __name__ == '__main__':
    # parse arguments
    # torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=int, default=[0], nargs='+')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--scale_factor', type=float, default=1.0)
    parser.add_argument('--if_superglue', type=str, default=False)
    parser.add_argument('--if_loftr', type=str, default=False)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    param = parser.parse_args()
    param.cur_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if param.config is not None:
        with open(param.config, 'r') as f:
            yaml_dict = yaml.safe_load(f)
            for k, v in yaml_dict.items():
                param.__dict__[k] = v
    # if param.if_superglue[0] == "F":
    #     param.if_superglue = False
    # if param.if_loftr[0] == "F":
    #     param.if_loftr = False
    param.if_aspanformer = False
    param.if_pdc_plus = False
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(param.master_port)
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    torch.cuda.empty_cache()
    # total_memory = torch.cuda.get_device_properties(0).total_memory
    # tmp_tensor = torch.empty(int(total_memory * 0.699), dtype=torch.int8, device='cuda')
    # pynvml.nvmlInit()
    # handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 0表示显卡标号
    # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    # print(meminfo.used / 1024 ** 2)

    # initialize random seed
    torch.manual_seed(param.seed)
    np.random.seed(param.seed)
    random.seed(param.seed)
    visible_devices = ''
    for id in param.gpu_ids:
        visible_devices += '{},'.format(id)
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
    param.saver=Saver(save_dir=os.path.join(param.save_dir, param.cur_time))
    # define logger and saver
    # handle data
    # test_dataset = CocoData(is_train=False, data_path=param.data_path, seed=1034)
    # if param.loss_type=="homography":
    #     test_dataset = CocoData(is_train=False, data_path=param.data_path, seed=1034)
    # else:
    #     if param.mode=="train" or param.mode=="test":
    #         # test_dataset = MegaData(label_path=param.label_path, label_reverse_path=param.label_reverse_path, pairs_path=param.pairs_path, is_train=False,
    #         #                     data_path=param.data_path)
    #         test_dataset = MegaData3(label_path=param.label_path, label_reverse_path=param.label_reverse_path, pairs_path=param.pairs_path, is_train=False,
    #                             data_path=param.origin_data_path)
    #         # test_dataset = ScanNet(data_path=param.Scannet_path, pairs_path=param.Scannet_parameter_path, is_train=False)
    #         # test_dataset = ScanNet(data_path=param.Scannet_path, pairs_path=param.Scannet_parameter_path, is_train=False)
    #     if param.mode=="refine" or param.mode=="refine_test":
    #         test_dataset = MegaData2(label_path=param.label_path, label_reverse_path=param.label_reverse_path, pairs_path=param.pairs_path, is_train=False,
    #                             data_path=param.origin_data_path, if_gray=(param.if_superglue or param.if_loftr or param.if_aspanformer), scale_factor=param.scale_factor)
            # test_dataset = ScanNet(data_path=param.Scannet_path, pairs_path=param.Scannet_parameter_path, is_train=False, if_refine=True)
    # prefetch_loader = data_prefetcher(train_loader, device)
    # param.test_dataset = test_dataset
    # test_loader = data_prefetcher(test_loader, device)

    # param.train_dataset = train_dataset
    # train
    if param.mode == 'train':
        param.local_rank = int(os.environ['LOCAL_RANK'])
        init_distributed_mode(param)
        model = SuperGlue_new()
        param.model = model
        device = torch.device('cuda:0' if torch.cuda.is_available() and len(param.gpu_ids) > 0 else 'cpu')
        param.train_dataset = MegaData3(label_path=param.label_path, label_reverse_path=param.label_reverse_path, pairs_path=param.pairs_path, is_train=True,
                            data_path=param.origin_data_path, seed=param.seed)
        if param.checkpoint is not None:
            assert os.path.exists(param.checkpoint)
            model_dict = torch.load(param.checkpoint, map_location= "cuda:"+str(param.local_rank))
            from collections import OrderedDict
            new_model_dict = model.state_dict()
            for k, v in model_dict.items():
                name = k[7:]
                # if str(name[:14]) != "deformable_gnn":
                new_model_dict[name] = v
            eval_flag = param.if_eval
            model.load_state_dict(new_model_dict)
        # mp.spawn(train, nprocs=np.array(param.gpu_ids).size, args=(param,))
        train(param)
    # test
    if param.mode == 'test':
        model = SuperGlue_new()
        param.model = model
        device = torch.device('cuda:0' if torch.cuda.is_available() and len(param.gpu_ids) > 0 else 'cpu')
        assert param.checkpoint is not None
        assert os.path.exists(param.checkpoint)
        model_dict = torch.load(param.checkpoint, map_location=device)
        from collections import OrderedDict
        new_model_dict = model.state_dict()
        test_dataset = MegaData3(label_path=param.label_path, label_reverse_path=param.label_reverse_path, pairs_path=param.pairs_path, is_train=False,
                                    data_path=param.origin_data_path)
        test_loader = DataLoader(test_dataset, batch_size=1,
                            num_workers=param.num_workers, drop_last=False)
        for k, v in model_dict.items():
            name = k[7:]
            if str(name[:7]) != "evaluat":
                new_model_dict[name] = v
        model.load_state_dict(new_model_dict)
        model = model.cuda()
        test(model, test_loader, param, device, 0)
