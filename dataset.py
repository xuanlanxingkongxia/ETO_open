import imp
from torch.utils.data.dataset import Dataset
import torch
import os
import numpy as np
from my_third_party.DenseMatching.demos.utils import scale_intrinsics
from utils.utils import Get_all_pairs, Get_pairs, Get_cameras
import cv2
from utils.utils import Resize_img, Resize_depth
import copy
from utils.hpatches_utils import hpatches_get_data
from PIL import Image
from PIL import Image
from datasets.scannet_extract import create_megadepth_label_refine
from datasets.scannet_extract import create_scannet_label_refine, scannet_get_pairs, create_scannet_label, get_scannet_pose
import h5py
from os import path as osp
import io
import torch.nn.functional as F
# import concurrent.futures.thread.ThreadPoolExecutor as ThreadPoolExecutor
# from scipy.spatial.transform import Rotation as R

def img_show(left, right, label, sequence):
    height1, width1, _ = left.shape
    height2, width2, _ = right.shape
    h = max(height1, height2)
    left = cv2.copyMakeBorder(left, 0, h - height1, 0, 0, cv2.BORDER_REPLICATE)
    right = cv2.copyMakeBorder(right, 0, h - height2, 0, 0, cv2.BORDER_REPLICATE)
    img = cv2.hconcat([left, right])
    patch_size = 1
    for j in range(height1//2//patch_size):
        j2 = j
        for k in range(width1//2//patch_size):
            k2 = k
            num = j2 * (width1 // 2 // patch_size) + k2
            if label[num, 0] > -0.1 and j%8==0 and k % 8==0:
                cv2.line(img, (int((2 * k2 + 1) * patch_size), int((2 * j2 + 1) * patch_size)),
                         (int(label[num, 0] + width1),
                          int(label[num, 1])),
                         [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)])
                # cv2.circle(left, (int((2 * k2 + 1) * patch_size), int((2 * j2 + 1) * patch_size)), 
                #     [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)])
    cv2.imwrite("/home/nijunjie/PATS/test/" + str(sequence) + ".jpg", img)




def resampling_mega(pairs_path, label_path, label_reverse_path, type):
    pairs, _ = Get_all_pairs(pairs_path, type)
    thread_num = pairs.shape[0]
    label = np.zeros((pairs.shape[0], 300, 3))
    reverse_label = np.zeros((pairs.shape[0], 300, 3))
    for i in range(thread_num):
        the_pair = pairs[i]
        label[i] = np.load(label_path + the_pair[0] + "/" + the_pair[1] + ".npy")
        reverse_label[i] = np.load(label_reverse_path + the_pair[0] + "/" + the_pair[1] + ".npy")
        if i%10000 == 0:
            print(i)
    matching_num = (label[:, :, 0] > -0.01).astype(np.int).sum(1)
    matching_reverse_num = (reverse_label[:, :, 0] > -0.01).astype(np.int).sum(1)
    nomatching_num = (label[:, :, 0] < -100).astype(np.int).sum(1)
    nomatching_reverse_num = (reverse_label[:, :, 0] < -100).astype(np.int).sum(1)
    criterion1 = np.logical_and(matching_num >= 20, matching_reverse_num >=1)
    criterion2 = np.logical_and(nomatching_num >= 10, nomatching_reverse_num >= 1)
    criterion = np.logical_and(criterion1, criterion2)
    np.save(pairs_path + "sampling2_" + type, criterion)


def save_result(result, all_pairs, F_gt, save_path):
    whole_data = []
    print(np.array(result).shape, F_gt.shape[0])
    for i in range(F_gt.shape[0]):
        left_medium_path = all_pairs[i][0]
        right_medium_path = all_pairs[i][0]
        left_name = all_pairs[i][3]
        right_name = all_pairs[i][2]
        if result[i] != -1:
            data = {
                "left_path": left_medium_path,
                "left_name": left_name,
                "right_path": right_medium_path,
                "right_name": right_name,
                "matches_l": result[i]["matches_l"],
                "matches_r": result[i]["matches_r"],
                "matches_num": result[i]["matches_l"].shape[0],
                "scale": result[i]["scale"],
                "F_gt": F_gt[i]
            }
        else:
            data= -1
        whole_data.append(data)
    np.save(save_path + "test_result", np.array(whole_data))

class data_prefetcher():
    def __init__(self, loader, device):
        self.device = device
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.P, self.the_left, self.the_right, self.depth_left, self.depth_right, self.left_path, self.right_path = next(self.loader)
        except StopIteration:
            self.the_label = None
            self.the_left = None
            self.the_right = None
            self.the_reverse_label = None
            return
        with torch.cuda.stream(self.stream):
            self.P = self.P.to(device=self.device, non_blocking=True)
            self.the_left = self.the_left.to(device=self.device, non_blocking=True)
            self.the_right = self.the_right.to(device=self.device, non_blocking=True)
            self.depth_left = self.depth_left.to(device=self.device, non_blocking=True)
            self.depth_right = self.depth_right.to(device=self.device, non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        P, the_left, the_right, depth_left, depth_right, left_path, right_path = self.P, self.the_left, self.the_right, self.depth_left, self.depth_right, self.left_path, self.right_path
        self.preload()
        return P, the_left, the_right, depth_left, depth_right, left_path, right_path


class MegaData(Dataset):
    def __init__(self, data_path, label_path, label_reverse_path, pairs_path, seed=3024, is_train=True):
        if is_train:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            print(seed)
            # resampling_mega(pairs_path, label_path, label_reverse_path, "train")
            # sampling = np.load(pairs_path + "sampling_train.npy")
            # pairs, F_list = Get_all_pairs(pairs_path, 'train')
            # pairs = np.load(pairs_path + "mega_data_pairs_train.npy")
            # F_list = np.load(data_path + "train_F_list.npy")
            pairs = np.load(pairs_path + "mega_data_pairs_train.npy")
            F_list = np.load(pairs_path + "train_F_list.npy")
            # print(sampling.astype(int).sum())
            # pairs = pairs[sampling]
            sequence = torch.randperm(int(pairs.shape[0]))
            data_pairs = pairs[sequence]
            # F_list = F_list[sampling]
            # np.save("train_F_list", np.array(F_list))
            F_list = F_list[sequence]            # resampling_mega(pairs_path, label_path, label_reverse_path, "train")

        else:
            # pairs, F_list = Get_all_pairs(pairs_path, 'test')
            pairs = np.load(pairs_path + "mega_data_pairs_test.npy")
            F_list = np.load(pairs_path + "test_F_list.npy")
            # resampling_mega(pairs_path, label_path, label_reverse_path, "test")
            # sampling = np.load(pairs_path + "sampling_test.npy")
            # print(sampling.astype(int).sum())
            # pairs = pairs[sampling]
            sequence = torch.randperm(int(pairs.shape[0]))
            data_pairs = pairs[sequence]
            # F_list = F_list[sampling]
            # np.save("test_F_list", np.array(F_list))
            F_list = F_list[sequence]
            self.all_pairs = data_pairs[:int(0.01 * pairs.shape[0])]
            self.F_list = F_list[:int(0.01 * pairs.shape[0])]
            self.transposed_F_list = self.F_list.transpose((0, 2, 1))
        self.data_path = data_path
        self.label_path = label_path
        self.label_reverse_path = label_reverse_path

    def __getitem__(self, item):
        the_pair = self.all_pairs[item]
        the_label_position = np.load(self.label_path + the_pair[0] + "/" + the_pair[1] + ".npy")
        the_label_position_reverse = np.load(self.label_reverse_path + the_pair[0] + "/" + the_pair[1] + ".npy")
        the_label_epopolar = self.F_list[item]
        the_label_epipolar_reverse = self.transposed_F_list[item]
        the_label = np.concatenate([the_label_position, the_label_epopolar], axis=0)
        the_label_reverse = np.concatenate([the_label_position_reverse, the_label_epipolar_reverse], axis=0)
        left_path = self.data_path + the_pair[0] + "/" + the_pair[3]
        right_path = self.data_path + the_pair[0] + "/" + the_pair[2]
        the_left = np.asfarray(Image.open(left_path))
        the_right = np.asfarray(Image.open(right_path))
        return the_label, the_label_reverse, the_left, the_right, left_path, right_path

    def __len__(self):
        return self.all_pairs.shape[0]

class MegaData2(Dataset):
    def __init__(self, data_path, label_path, label_reverse_path, pairs_path, seed=7787, is_train=True, if_gray=False, scale_factor=1):
        if is_train:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            pairs, F_list = Get_all_pairs(pairs_path, 'train')
            sampling = np.load(pairs_path + "sampling_train.npy")
            print(sampling.astype(int).sum())
            print(seed)
            pairs = pairs[sampling]
            sequence = torch.randperm(int(pairs.shape[0]))
            data_pairs = pairs[sequence]
            # F_list = F_list[sampling]
            # F_list = F_list[sequence]
            self.all_pairs = data_pairs[:int(0.05 * pairs.shape[0])]
            # self.F_list = F_list[:int(0.1 * pairs.shape[0])]
            # self.transposed_F_list = self.F_list.transpose((0, 2, 1))
            self.imgs = Get_cameras(pairs_path, data_path, "train", if_origin=False)
        else:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            # 去除了loftr的训练集之后的测试集
            pairs, F_list = Get_all_pairs(pairs_path, 'test')
            # 完整的测试集
            # pairs, F_list = Get_all_pairs(pairs_path, 'test_full')
            pairs_back, _ = Get_all_pairs(pairs_path, 'test2')
            # resampling_mega(pairs_path, label_path, label_reverse_path, "test")
            # 完整的测试集
            # sampling = np.load(pairs_path + "sampling_test.npy")
            # 去除了loftr的训练集之后的测试集
            sampling = np.load(pairs_path + "sampling2_test.npy")
            print(sampling.astype(int).sum())

            data_pairs = pairs[sampling]
            sequence = torch.randperm(int(data_pairs.shape[0]))
            data_pairs = data_pairs[sequence]
            self.all_pairs = data_pairs[:1000]
            # sequence2 = torch.randperm(1000)
            # self.all_pairs = self.all_pairs[sequence2]

            self.back_pairs = pairs_back[int(0.00 * data_pairs.shape[0]):int(0.1 * data_pairs.shape[0])]
            # self.all_pairs = data_pairs[int(0.00 * data_pairs.shape[0]):int(0.01 * data_pairs.shape[0])]
            # 3, 4, 14, 16, 27
            # self.all_pairs = data_pairs[9:10]

            for i in range(1):
                the_left = cv2.imread(data_path + data_pairs[11][0] + "/imgs/" + data_pairs[11][3])
                the_left = Resize_img(the_left, np.array([1024, 768]))
                cv2.imwrite("/home/nijunjie/FDesp_new/test_ply/origin_left.jpg", the_left)
                the_right = cv2.imread(data_path + data_pairs[11][0] + "/imgs/" + data_pairs[11][2])
                the_right = Resize_img(the_right, np.array([1024, 768]))
                cv2.imwrite("/home/nijunjie/FDesp_new/test_ply/origin_right.jpg", the_right)

            # F_list = F_list[sampling]
            # F_list = F_list[sequence]
            # self.F_list = F_list[int(0.00 * data_pairs.shape[0]):int(0.01 * data_pairs.shape[0])]
            # self.transposed_F_list = self.F_list.transpose((0, 2, 1))
            self.imgs = Get_cameras(pairs_path, data_path, "test", if_origin=False)
        self.pairs_path = pairs_path
        self.data_path = data_path
        self.label_path = label_path
        self.label_reverse_path = label_reverse_path
        self.if_gray=if_gray
        self.scale_factor = scale_factor
        if scale_factor > 1.0:
            self.if_change_left = True
        else:
            self.if_change_left = False
        self.is_train=is_train

    def __getitem__(self, item):
        w = 640
        h = 480
        the_pair = self.all_pairs[item]
        left_path = self.data_path + the_pair[0] + "/imgs/" + the_pair[3]
        left_path_depth = self.data_path + the_pair[0] + "/depths/" + the_pair[3].split('.')[0]
        right_path = self.data_path + the_pair[0] + "/imgs/" + the_pair[2]
        right_path_depth = self.data_path + the_pair[0] + "/depths/" + the_pair[2].split('.')[0]
        if not(self.is_train):
            background_pair = self.back_pairs[np.random.randint(0, self.back_pairs.shape[0])]
            background_path = self.data_path + background_pair[0] + "/imgs/" + background_pair[3]
        if self.if_gray:
            the_left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
            the_right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
            the_background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
            the_left = Resize_depth(the_left, np.array([w, h]))
            # the_right = Resize_depth(the_right, np.array([640, 480]))
            the_background = Resize_depth(the_background, np.array([int(w * max(self.scale_factor, 1)), int(h * max(self.scale_factor, 1))]))
            the_right = Resize_depth(the_right, np.array([int(w * self.scale_factor), int(h * self.scale_factor)]))
        else:
            the_left = cv2.imread(left_path)[:, :, [2, 1, 0]]
            the_right = cv2.imread(right_path)[:, :, [2, 1, 0]]
            the_left = Resize_img(the_left, np.array([w, h]))
            the_right = Resize_img(the_right, np.array([int(w * self.scale_factor), int(h * self.scale_factor)]))
            if not(self.is_train):
                the_background = cv2.imread(background_path)[:, :, [2, 1, 0]]
                the_background = Resize_img(the_background, np.array([int(w * max(self.scale_factor, 1)), int(h * max(self.scale_factor, 1))]))

        # left_depth = Resize_depth(np.asfarray(h5py.File(left_path_depth + ".h5", "r")['depth']).astype(float), np.array([640, 480]))
        # right_depth = Resize_depth(np.asfarray(h5py.File(right_path_depth + ".h5", "r")['depth']).astype(float), np.array([640, 480]))

        h_l2, w_l2 = the_left.shape[:2]
        h_r2, w_r2 = the_right.shape[:2]
        max_width = max(w_l2, w_r2)
        max_height = max(h_l2, h_r2)
        # the_left = cv2.copyMakeBorder(the_left, 0, max_height - h_l2, 0, max_width - w_l2, cv2.BORDER_CONSTANT, None, 0)
        # the_right = cv2.copyMakeBorder(the_right, 0, max_height - h_r2, 0, max_width - w_r2, cv2.BORDER_CONSTANT, None, 0)
        right_depth = Resize_depth(np.asfarray(h5py.File(right_path_depth + ".h5", "r")['depth']).astype(float), np.array([int(w * self.scale_factor), int(h * self.scale_factor)]))
        left_depth = Resize_depth(np.asfarray(h5py.File(left_path_depth + ".h5", "r")['depth']).astype(float), np.array([w, h]))
        l = self.imgs[self.data_path + the_pair[0] + "/imgs/" + the_pair[3]]
        r = self.imgs[self.data_path + the_pair[0] + "/imgs/" + the_pair[2]]
        intrinsic_right = np.eye(4)
        intrinsic_right[:3, :3] = scale_intrinsics(r["K"].astype(float)[:3, :3], [float(w) / w_r2 , float(h) / h_r2])
        intrinsic_left = copy.copy(l["K"].astype(float))
        if not(self.is_train):
            if self.if_change_left:
                intrinsic_left[:2, 2] += np.asarray([(max_width - w_l2)//2, (max_height - h_l2)//2])
                left_depth = cv2.copyMakeBorder(left_depth, (max_height - h_l2)//2, (max_height - h_l2)//2, (max_width - w_l2)//2, (max_width - w_l2)//2, cv2.BORDER_CONSTANT, None, -1)
                the_right = cv2.copyMakeBorder(the_right, 0, max_height // 32 * 32 + int(max_height%32!=0) * 32 - h_r2, 0, max_width - w_r2, cv2.BORDER_CONSTANT, None, 0)
                the_background[(max_height - h_l2)//2:(max_height - h_l2)//2 + h, (max_width - w_l2)//2: (max_width - w_l2)//2 + w] = the_left
                h_l2, w_l2 = the_background.shape[:2]
                the_background = cv2.copyMakeBorder(the_background, 0, (max_height // 32 + int(max_height%32!=0)) * 32 - h_l2, 0, max_width - w_l2, cv2.BORDER_CONSTANT, None, 0)
                the_left = the_background
                left_depth = cv2.copyMakeBorder(left_depth, 0, max_height // 32 * 32 + int(max_height%32!=0) * 32  - h_l2, 0, max_width - w_l2, cv2.BORDER_CONSTANT, None, 0)
                right_depth = cv2.copyMakeBorder(right_depth, 0, max_height // 32 * 32 + int(max_height%32!=0) * 32  - h_r2, 0, max_width - w_r2, cv2.BORDER_CONSTANT, None, 0)
            else:
                the_background[(max_height - h_r2)//2:(max_height - h_r2)//2 + the_right.shape[0], (max_width - w_r2)//2: (max_width - w_r2)//2  + the_right.shape[1]] = the_right
                the_right = the_background
                intrinsic_right[:2, 2] += np.asarray([(max_width - w_r2)//2, (max_height - h_r2)//2])
                right_depth = cv2.copyMakeBorder(right_depth, (max_height - h_r2)//2, (max_height - h_r2)//2, (max_width - w_r2)//2, (max_width - w_r2)//2, cv2.BORDER_CONSTANT, None, -1)
        the_label, the_label_reverse = create_megadepth_label_refine(intrinsic_left, intrinsic_right, 
            left_depth, right_depth, l["P"].astype(float), r["P"].astype(float))
        return the_label, the_label_reverse, the_left, the_right, left_path, right_path

    def __len__(self):
        return self.all_pairs.shape[0]

def create_undistort_param(param_path, save_path):
    scene_name = np.sort(np.array(os.listdir(save_path)))
    for i in np.array(range(scene_name.shape[0] - scene_name.shape[0] + 1)) + 164:
        dense_name = np.sort(np.array(os.listdir(save_path + scene_name[i])))
        for j in range(dense_name.shape[0]):
            print(i, j)
            old_cam_path = save_path + scene_name[i] + "/" + dense_name[j] + "/img_cam.txt"
            f_old_cam = open(old_cam_path, "r").readlines()
            save_cam_path = save_path + scene_name[i] + "/" + dense_name[j] + "/img_cam_new.txt"
            f_save = open(save_cam_path, 'w')
            new_camera_path = param_path + scene_name[i] + "/" + dense_name[j] + "/sparse-txt/cameras.txt"
            if not (os.path.exists(new_camera_path)):
                continue
            f_new_camera = open(new_camera_path, "r").readlines()
            new_cams = {}
            new_img_path = param_path + scene_name[i] + "/" + dense_name[j] + "/sparse-txt/images.txt"
            f_new_img = open(new_img_path).readlines()
            new_imgs = {}
            for new_cam_line in f_new_camera[3:]:
                new_cam_line_split = new_cam_line[:-1].split()
                new_cam = {'w': new_cam_line_split[2], 'h': new_cam_line_split[3], 'f_x': new_cam_line_split[4], 'f_y': new_cam_line_split[5],
                    'c_x': new_cam_line_split[6], 'c_y': new_cam_line_split[7]}
                new_cams[new_cam_line_split[0]] = new_cam
            for k, new_img_line in enumerate(f_new_img[4:]):
                if k % 2 == 0:
                    new_img_line_split = new_img_line[:-1].split()
                    r = np.array(R.from_quat(np.array(new_img_line_split[1:5]).astype(float)).as_matrix())
                    t = np.array(new_img_line_split[5:8]).reshape(3, 1)
                    p_matrix = np.concatenate((r, t), axis=1).reshape(-1)
                    new_img = {'p': p_matrix, 'k_id': new_img_line_split[8]}
                    new_imgs[new_img_line_split[9]] = new_img
            count = 0
            for old_cam_line in f_old_cam:
                old_cam_line_split = old_cam_line[:-1].split()
                name = old_cam_line_split[0]
                if name in new_imgs:
                    the_img = new_imgs[name]
                    the_camera = new_cams[the_img['k_id']]
                    output = name + " " + str(int(the_camera['w'])) + " " + str(int(the_camera['h'])) + " " + str(the_camera['f_x']) + " " + str(the_camera['f_y']) + " " + \
                        str(the_camera['c_x']) + " " + str(the_camera['c_y']) + " " + " ".join(str(i) for i in the_img['p']) + "\n"
                    f_save.write(output)
                else:
                    count += 1
            print(count)
            f_save.close()


class MegaData3(Dataset):
    def __init__(self, data_path, label_path, label_reverse_path, pairs_path, seed=2048, is_train=True):
        if is_train:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            pairs = np.load(pairs_path + "megadepth_train.npy")
            sequence = torch.randperm(int(pairs.shape[0]))
            data_pairs = pairs[sequence]
            self.all_pairs = data_pairs[:int(pairs.shape[0] * 0.15)]
            self.imgs = Get_cameras(pairs_path, data_path, "train", if_origin=False)
            self.setting = "train"
            # self.image = {}
            # self.depth = {}
            # img = set()
            # depth = set()
            # x = 0
            # for pair in pairs:
            #     img.add(str(data_path + "img/" + pair[0] + "/" + pair[3]))
            #     img.add(str(data_path + "img/" + pair[0] + "/" + pair[2]))
            #     depth.add(str(data_path + "depth/" + pair[0] + "/" + pair[3].split('.')[0]))
            #     depth.add(str(data_path + "depth/" + pair[0] + "/" + pair[3].split('.')[0]))
            # for i_path in img:
            #     x += 1
            #     img_path = i_path + ".npy"
            #     self.image[i_path] = np.load(img_path)
            #     # i = cv2.imread(data_path + i_path)[:, :, [2, 1, 0]]
            #     # h_l, w_l = i.shape[:2]
            #     # max_shape = max(h_l, w_l)
            #     # size_l = 1600.0 / max_shape
            #     # # the_left = Resize_img(the_left, np.array([int(w_l * size_l), int(h_l * size_l)]))
            #     # i = Resize_img(i, np.array([640, 480]))
            #     # h_l2, w_l2 = i.shape[:2]
            #     # i = i[:h_l2//32*32, :w_l2//32*32]
            #     # if not os.path.exists("/mnt/nas_7/datasets/nijunjie_changed_megadepth/img/" + i_path.split('/')[0]):
            #     #     os.mkdir("/mnt/nas_7/datasets/nijunjie_changed_megadepth/img/" + i_path.split('/')[0])
            #     # if not os.path.exists("/mnt/nas_7/datasets/nijunjie_changed_megadepth/img/" + i_path.split('/')[0] + "/" + i_path.split('/')[1]):
            #     #     os.mkdir("/mnt/nas_7/datasets/nijunjie_changed_megadepth/img/" + i_path.split('/')[0] + "/" + i_path.split('/')[1])
            #     # np.save("/mnt/nas_7/datasets/nijunjie_changed_megadepth/img/" + i_path.split('/')[0] + "/" + i_path.split('/')[1] + "/" + i_path.split('/')[3] , i)
            #     if x % 1000==0:
            #         print(x)
            # for i_path in depth:
            #     x += 1
            #     depth_path = i_path + ".npy"
            #     self.depth[i_path] = np.load(depth_path)
            #     # i = Resize_depth(np.asfarray(h5py.File(data_path + i_path + ".h5", "r")['depth']).astype(float), np.array([640, 480]))[:h_l2//32*32, :w_l2//32*32]
            #     # if not os.path.exists("/mnt/nas_7/datasets/nijunjie_changed_megadepth/depth/" + i_path.split('/')[0]):
            #     #     os.mkdir("/mnt/nas_7/datasets/nijunjie_changed_megadepth/depth/" + i_path.split('/')[0])
            #     # if not os.path.exists("/mnt/nas_7/datasets/nijunjie_changed_megadepth/depth/" + i_path.split('/')[0] + "/" + i_path.split('/')[1]):
            #     #     os.mkdir("/mnt/nas_7/datasets/nijunjie_changed_megadepth/depth/" + i_path.split('/')[0] + "/" + i_path.split('/')[1])
            #     # np.save("/mnt/nas_7/datasets/nijunjie_changed_megadepth/depth/" + i_path.split('/')[0] + "/" + i_path.split('/')[1] + "/" + i_path.split('/')[3].split('.')[0], i)
            #     if x % 1000==0:
            #         print(x)
        else:
            pairs = np.load(pairs_path + "megadepth_test.npy")
            sequence = torch.randperm(int(pairs.shape[0]))
            data_pairs = pairs[sequence]
            self.all_pairs = data_pairs[0:1000]
            self.imgs = Get_cameras(pairs_path, data_path, "test", if_origin=False)
            self.setting = "test"
            # img = set()
            # depth = set()
            # for pair in pairs:
            #     img.add(str(pair[0] + "/imgs/" + pair[3]))
            #     img.add(str(pair[0] + "/imgs/" + pair[2]))
            #     depth.add(str(pair[0] + "/depths/" + pair[3].split('.')[0]))
            #     depth.add(str(pair[0] + "/depths/" + pair[2].split('.')[0]))
            # print("start")
            # x = 0
            # for i_path in img:
            #     x += 1
            #     # i = cv2.imread(data_path + i_path)[:, :, [2, 1, 0]]
            #     # h_l, w_l = i.shape[:2]
            #     # max_shape = max(h_l, w_l)
            #     # size_l = 1600.0 / max_shape
            #     # # the_left = Resize_img(the_left, np.array([int(w_l * size_l), int(h_l * size_l)]))
            #     # i = Resize_img(i, np.array([640, 480]))
            #     # h_l2, w_l2 = i.shape[:2]
            #     # i = i[:h_l2//32*32, :w_l2//32*32]
            #     # if not os.path.exists("/mnt/nas_7/datasets/nijunjie_changed_megadepth/img/" + i_path.split('/')[0]):
            #     #     os.mkdir("/mnt/nas_7/datasets/nijunjie_changed_megadepth/img/" + i_path.split('/')[0])
            #     # if not os.path.exists("/mnt/nas_7/datasets/nijunjie_changed_megadepth/img/" + i_path.split('/')[0] + "/" + i_path.split('/')[1]):
            #     #     os.mkdir("/mnt/nas_7/datasets/nijunjie_changed_megadepth/img/" + i_path.split('/')[0] + "/" + i_path.split('/')[1])
            #     # np.save("/mnt/nas_7/datasets/nijunjie_changed_megadepth/img/" + i_path.split('/')[0] + "/" + i_path.split('/')[1] + "/" + i_path.split('/')[3] , i)
            #     if x % 100==0:
            #         print(x)
            # for i_path in depth:
            #     x += 1
            #     # i = Resize_depth(np.asfarray(h5py.File(data_path + i_path + ".h5", "r")['depth']).astype(float), np.array([640, 480]))[:h_l2//32*32, :w_l2//32*32]
            #     # if not os.path.exists("/mnt/nas_7/datasets/nijunjie_changed_megadepth/depth/" + i_path.split('/')[0]):
            #     #     os.mkdir("/mnt/nas_7/datasets/nijunjie_changed_megadepth/depth/" + i_path.split('/')[0])
            #     # if not os.path.exists("/mnt/nas_7/datasets/nijunjie_changed_megadepth/depth/" + i_path.split('/')[0] + "/" + i_path.split('/')[1]):
            #     #     os.mkdir("/mnt/nas_7/datasets/nijunjie_changed_megadepth/depth/" + i_path.split('/')[0] + "/" + i_path.split('/')[1])
            #     # np.save("/mnt/nas_7/datasets/nijunjie_changed_megadepth/depth/" + i_path.split('/')[0] + "/" + i_path.split('/')[1] + "/" + i_path.split('/')[3].split('.')[0], i)
            #     if x % 100==0:
            #         print(x)
        self.pairs_path = pairs_path
        self.data_path = data_path
        self.label_path = label_path
        self.label_reverse_path = label_reverse_path

    def __getitem__(self, item):
        the_pair = self.all_pairs[item]
        # left_path = self.data_path + "img/" + the_pair[0] + "/" + the_pair[3]
        # right_path = self.data_path + "img/" + the_pair[0] + "/" + the_pair[2]
        # the_left = self.image[left_path]
        # the_right = self.image[right_path]
        # left_depth = self.depth[self.data_path + "depth/" + the_pair[0] + "/" + the_pair[3].split('.')[0]]
        # right_depth = self.depth[self.data_path + "depth/" + the_pair[0] + "/" + the_pair[2].split('.')[0]]

        left_path = self.data_path + "img/" + the_pair[0] + "/" + the_pair[3] + ".npy"
        right_path = self.data_path + "img/" + the_pair[0] + "/" + the_pair[2] + ".npy"
        the_left = np.load(left_path)
        the_right = np.load(right_path)
        left_depth = np.load(self.data_path + "depth/" + the_pair[0] + "/" + the_pair[3].split('.')[0] + ".npy")
        right_depth = np.load(self.data_path + "depth/" + the_pair[0] + "/" + the_pair[2].split('.')[0] + ".npy")


        # left_path = self.data_path + the_pair[0] + "/imgs/" + the_pair[3]
        # left_path_depth = self.data_path + the_pair[0] + "/depths/" + the_pair[3].split('.')[0]
        # right_path = self.data_path + the_pair[0] + "/imgs/" + the_pair[2]
        # right_path_depth = self.data_path + the_pair[0] + "/depths/" + the_pair[2].split('.')[0]           
        # the_left = cv2.imread(left_path)[:, :, [2, 1, 0]]
        # h_l, w_l = the_left.shape[:2]
        # max_shape = max(h_l, w_l)
        # size_l = 1600.0 / max_shape
        # # the_left = Resize_img(the_left, np.array([int(w_l * size_l), int(h_l * size_l)]))
        # the_left = Resize_img(the_left, np.array([640, 480]))
        # h_l2, w_l2 = the_left.shape[:2]
        # the_left = the_left[:h_l2//32*32, :w_l2//32*32]
        # the_right = cv2.imread(right_path)[:, :, [2, 1, 0]]
        # h_r, w_r = the_right.shape[:2]
        # max_shape = max(h_r, w_r)
        # size_r = 1600.0 / max_shape
        # # the_right = Resize_img(the_right, np.array([int(w_r * size_r), int(h_r * size_r)]))
        # the_right = Resize_img(the_right, np.array([640, 480]))
        # h_r2, w_r2 = the_right.shape[:2]
        # the_right = the_right[:h_r2//32*32, :w_r2//32*32]
        # max_width = max(w_l2//32*32, w_r2//32*32)
        # max_height = max(h_l2//32*32, h_r2//32*32)
        # max_width = 1024
        # max_height = 1024
        # the_left = cv2.copyMakeBorder(the_left, 0, max_height - h_l2//32*32, 0, max_width - w_l2//32*32, cv2.BORDER_CONSTANT, None, 0)
        # the_right = cv2.copyMakeBorder(the_right, 0, max_height - h_r2//32*32, 0, max_width - w_r2//32*32, cv2.BORDER_CONSTANT, None, 0)
        # left_depth = Resize_depth(np.asfarray(h5py.File(left_path_depth + ".h5", "r")['depth']).astype(float), np.array([int(w_l * size_l), int(h_l * size_l)]))[:h_l2//32*32, :w_l2//32*32]
        # right_depth = Resize_depth(np.asfarray(h5py.File(right_path_depth + ".h5", "r")['depth']).astype(float), np.array([int(w_r * size_r), int(h_r * size_r)]))[:h_r2//32*32, :w_r2//32*32]
        # left_depth = Resize_depth(np.asfarray(h5py.File(left_path_depth + ".h5", "r")['depth']).astype(float), np.array([640, 480]))[:h_l2//32*32, :w_l2//32*32]
        # right_depth = Resize_depth(np.asfarray(h5py.File(right_path_depth + ".h5", "r")['depth']).astype(float), np.array([640, 480]))[:h_r2//32*32, :w_r2//32*32]
        # left_depth = cv2.copyMakeBorder(left_depth, 0, max_height - h_l2//32*32, 0, max_width - w_l2//32*32, cv2.BORDER_CONSTANT, None, 0)
        # right_depth = cv2.copyMakeBorder(right_depth, 0, max_height - h_r2//32*32, 0, max_width - w_r2//32*32, cv2.BORDER_CONSTANT, None, 0)
        # left_depth = np.asfarray(h5py.File(left_path_depth + ".h5", "r")['depth']).astype(float)[:h_l2//32*32, :w_l2//32*32]
        # right_depth = np.asfarray(h5py.File(right_path_depth + ".h5", "r")['depth']).astype(float)[:h_r2//32*32, :w_r2//32*32]
        # if not os.path.exists("/mnt/nas_7/datasets/nijunjie_changed_megadepth/img/" + the_pair[0].split('/')[0]):
        #     os.mkdir("/mnt/nas_7/datasets/nijunjie_changed_megadepth/img/" + the_pair[0].split('/')[0])
        # if not os.path.exists("/mnt/nas_7/datasets/nijunjie_changed_megadepth/img/" + the_pair[0]):
        #     os.mkdir("/mnt/nas_7/datasets/nijunjie_changed_megadepth/img/" + the_pair[0])
        # np.save("/mnt/nas_7/datasets/nijunjie_changed_megadepth/img/" + the_pair[0] + "/" + the_pair[3], the_left)
        # np.save("/mnt/nas_7/datasets/nijunjie_changed_megadepth/img/" + the_pair[0] + "/" + the_pair[2], the_right)
        # if not os.path.exists("/mnt/nas_7/datasets/nijunjie_changed_megadepth/depth/" + the_pair[0].split('/')[0]):
        #     os.mkdir("/mnt/nas_7/datasets/nijunjie_changed_megadepth/depth/" + the_pair[0].split('/')[0])
        # if not os.path.exists("/mnt/nas_7/datasets/nijunjie_changed_megadepth/depth/" + the_pair[0]):
        #     os.mkdir("/mnt/nas_7/datasets/nijunjie_changed_megadepth/depth/" + the_pair[0])
        # np.save("/mnt/nas_7/datasets/nijunjie_changed_megadepth/depth/" + the_pair[0] + "/" + the_pair[3].split('.')[0], left_depth)
        # np.save("/mnt/nas_7/datasets/nijunjie_changed_megadepth/depth/" + the_pair[0] + "/" + the_pair[2].split('.')[0], right_depth)
        l_pose = self.imgs[self.data_path + "img/" + the_pair[0] + "/" + the_pair[3] + ".npy"]
        r_pose = self.imgs[self.data_path + "img/" + the_pair[0] + "/" + the_pair[2] + ".npy"]
        # the_label, the_label_reverse = create_megadepth_label(l["K"].astype(float), r["K"].astype(float), 
        #     left_depth, right_depth, l["P"].astype(float), r["P"].astype(float))
        # img_show(the_left, the_right, the_label, item)
        P = r_pose["K"].astype(float).dot(r_pose["P"].astype(float)).dot(np.linalg.inv(l_pose["K"].astype(float).dot(l_pose["P"].astype(float))))
        return P, the_left, the_right, left_depth, right_depth, left_path, right_path

    def __len__(self):
        return self.all_pairs.shape[0]

class ScanNet(Dataset):
    def __init__(self, data_path, pairs_path, seed=3024, is_train=True, if_refine=False):
        if is_train:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            print(seed)
            pairs = scannet_get_pairs(pairs_path, "train")
            # sampling = np.ones([pairs.shape[0]]).astype(bool)
            # for i in range(pairs.shape[0]):
            #     the_pair = pairs[i]
            #     left_pose_path = data_path + "scans/" + the_pair[0] + "/pose/" + the_pair[1] + ".txt"
            #     right_pose_path = data_path + "scans/" + the_pair[0] + "/pose/" + str(int(the_pair[1]) + 50) + ".txt"
            #     left_pose = get_scannet_pose(left_pose_path)
            #     right_pose = get_scannet_pose(right_pose_path)
            #     if math.isinf(float(left_pose[0, 0])) or math.isinf(float(right_pose[0, 0])):
            #         sampling[i] = False
            #     if i%10000==0:
            #         print(i)
            # np.save(pairs_path + "sampling_train", sampling)
            sampling = np.load(pairs_path + "sampling_train.npy")
            pairs = pairs[sampling]
            sequence = torch.randperm(int(pairs.shape[0]))
            data_pairs = pairs[sequence]
            self.all_pairs = data_pairs[:int(0.03 * pairs.shape[0])]
        else:
            # pairs, F_list = Get_all_pairs(pairs_path, 'test')
            pairs = scannet_get_pairs(pairs_path, "test")
            # sampling = np.ones([pairs.shape[0]]).astype(bool)
            # for i in range(pairs.shape[0]):
            #     the_pair = pairs[i]
            #     left_pose_path = data_path + "scans/" + the_pair[0] + "/pose/" + the_pair[1] + ".txt"
            #     right_pose_path = data_path + "scans/" + the_pair[0] + "/pose/" + str(int(the_pair[1]) + 50) + ".txt"
            #     left_pose = get_scannet_pose(left_pose_path)
            #     right_pose = get_scannet_pose(right_pose_path)
            #     if math.isinf(float(left_pose[0, 0])) or math.isinf(float(right_pose[0, 0])):
            #         sampling[i] = False
            #     if i%10000==0:
            #         print(i)
            # np.save(pairs_path + "sampling_test", sampling)
            sampling = np.load(pairs_path + "sampling_test.npy")
            pairs = pairs[sampling]
            sequence = torch.randperm(int(pairs.shape[0]))
            data_pairs = pairs[sequence]
            self.all_pairs = data_pairs[int(0.0001 * pairs.shape[0]): int(0.0002 * pairs.shape[0])][[4, 8, 14, 17, 18, 43]][5:6]
            # if if_refine:
            #     self.manager = multiprocessing.Manager()
            #     self.images = self.manager.dict()
        self.data_path = data_path
        self.pairs_path = pairs_path
        self.intrinsics = np.load(pairs_path + "intrinsics.npy", allow_pickle=True).item()
        self.if_refine = if_refine
        self.is_train = is_train

    def __getitem__(self, item):
        the_pair = self.all_pairs[item]
        left_rgb_path = self.data_path + "scans/" + the_pair[0] + "/color2/" + the_pair[1] + ".jpg"
        right_rgb_path = self.data_path + "scans/" + the_pair[0] + "/color2/" + str(int(the_pair[1]) + 50) + ".jpg"
        left_pose_path = self.data_path + "scans/" + the_pair[0] + "/pose/" + the_pair[1] + ".txt"
        right_pose_path = self.data_path + "scans/" + the_pair[0] + "/pose/" + str(int(the_pair[1]) + 50) + ".txt"
        left_pose = np.linalg.inv(get_scannet_pose(left_pose_path).astype(float))
        right_pose = np.linalg.inv(get_scannet_pose(right_pose_path).astype(float))
        intrinsic_d = self.intrinsics[the_pair[0] + "_depth"].astype(float)
        if self.if_refine == False:
            left_depth_path = self.data_path + "scans/" + the_pair[0] + "/depth/" + the_pair[1] + ".png"
            right_depth_path = self.data_path + "scans/" + the_pair[0] + "/depth/" + str(int(the_pair[1]) + 50) + ".png"
            left_depth = cv2.imread(left_depth_path, -1).astype(float) / 1000.0
            right_depth = cv2.imread(right_depth_path, -1).astype(float) / 1000.0
            the_left = cv2.imread(left_rgb_path)[:, :, [2, 1, 0]]
            the_right = cv2.imread(right_rgb_path)[:, :, [2, 1, 0]]
            the_label, the_label_reverse = create_scannet_label(intrinsic_d, left_depth, right_depth, left_pose, right_pose)
            # img_show(the_left, the_right, the_label, item)
        else:
            left_depth_path = self.data_path + "scans/" + the_pair[0] + "/depth/" + the_pair[1] + ".png"
            right_depth_path = self.data_path + "scans/" + the_pair[0] + "/depth/" + str(int(the_pair[1]) + 50) + ".png"
            left_depth = cv2.imread(left_depth_path, -1).astype(float) / 1000.0
            right_depth = cv2.imread(right_depth_path, -1).astype(float) / 1000.0
            the_left = cv2.imread(left_rgb_path)[:, :, [2, 1, 0]]
            the_right = cv2.imread(right_rgb_path)[:, :, [2, 1, 0]]

            # max_shape = max(the_left.shape[0], the_left.shape[1])
            # scales = np.diag([640.0 / max_shape, 640.0 / max_shape, 1, 1])
            # intrinsic =  np.dot(scales, intrinsic_c)

            # size = 640
            # h_l, w_l = the_left.shape[:2]
            # max_shape_l = max(h_l, w_l)
            # size_l = size / max_shape_l
            # the_left = Resize_img(the_left, np.array([int(w_l * size_l), int(h_l * size_l)]))
            # # the_left = Resize_img(the_left, np.array([640, 480]))
            # h_l2, w_l2 = the_left.shape[:2]
            # h_r, w_r = the_right.shape[:2]
            # max_shape_r = max(h_r, w_r)
            # size_r = size / max_shape_r
            # the_right = Resize_img(the_right, np.array([int(w_r * size_r), int(h_r * size_r)]))
            # h_r2, w_r2 = the_right.shape[:2]
            # the_left = cv2.copyMakeBorder(the_left, 0, 480 - h_l2, 0, 640 - w_l2, cv2.BORDER_CONSTANT, None, 0)
            # the_right = cv2.copyMakeBorder(the_right, 0, 480 - h_r2, 0, 640 - w_r2, cv2.BORDER_CONSTANT, None, 0)

            the_label, the_label_reverse = create_scannet_label_refine(intrinsic_d, left_depth, right_depth, left_pose, right_pose)
            # if self.is_train == False:
            #     param1 = {'K': intrinsic_d, 'P': left_pose}
            #     param2 = {'K': intrinsic_d, 'P': right_pose}
            #     self.images[left_rgb_path] = param1
            #     self.images[right_rgb_path] = param2
        # the_left = cv2.resize(cv2.imread(left_rgb_path), (640, 480))
        # the_right = cv2.resize(cv2.imread(right_rgb_path), (640, 480))
        # img = cv2.hconcat([the_left, the_right])
        # print(img.shape)
        # img_show(img, the_label[:300], 0)
        return the_label, the_label_reverse, the_left, the_right, left_rgb_path, right_rgb_path

    def __len__(self):
        return self.all_pairs.shape[0]


class ScanNetDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path_list,
                 intrinsic_path,
                 mode='train',
                 min_overlap_score=0.4,
                 augment_fn=None,
                 pose_dir=None):
        """Manage one scene of ScanNet Dataset.
        Args:
            root_dir (str): ScanNet root directory that contains scene folders.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            intrinsic_path (str): path to depth-camera intrinsic file.
            mode (str): options are ['train', 'val', 'test'].
            augment_fn (callable, optional): augments images with pre-defined visual effects.
            pose_dir (str): ScanNet root directory that contains all poses.
                (we use a separate (optional) pose_dir since we store images and poses separately.)
        """
        super().__init__()
        self.root_dir = root_dir
        self.pose_dir = pose_dir if pose_dir is not None else root_dir
        self.mode = mode

        # prepare data_names, intrinsics and extrinsics(T)
        self.intrinsics = np.load(intrinsic_path)
        data_names = []
        for npz_path in npz_path_list:
            with np.load(npz_path) as data:
                data_name = data['name']
                if 'score' in data.keys() and mode not in ['val' or 'test']:
                    kept_mask = data['score'] > min_overlap_score
                    data_name = data_name[kept_mask]
                data_names.append(data_name)
        self.data_names = np.concatenate(data_names)
        np.save("~/FAST/scannet_indices/data_names", self.data_names)
        self.data_names = np.load("~/FAST/scannet_indices/data_names.npy")
        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None

    def __len__(self):
        return len(self.data_names)

    def _read_abs_pose(self, scene_name, name):
        pth = osp.join(self.pose_dir,
                       scene_name,
                       'pose', f'{name}.txt')
        return self.read_scannet_pose(pth)

    def read_scannet_pose(self, path):
        """ Read ScanNet's Camera2World pose and transform it to World2Camera.
        
        Returns:
            pose_w2c (np.ndarray): (4, 4)
        """
        cam2world = np.loadtxt(path, delimiter=' ')
        world2cam = np.linalg.inv(cam2world)
        return world2cam

    def _compute_rel_pose(self, scene_name, name0, name1):
        pose0 = self._read_abs_pose(scene_name, name0)
        pose1 = self._read_abs_pose(scene_name, name1)
        return np.matmul(pose1, np.linalg.inv(pose0))  # (4, 4)

    def load_array_from_s3(self,
        path, client, cv_type,
        use_h5py=False,
    ):
        byte_str = client.Get(path)
        try:
            if not use_h5py:
                raw_array = np.fromstring(byte_str, np.uint8)
                data = cv2.imdecode(raw_array, cv_type)
            else:
                f = io.BytesIO(byte_str)
                data = np.array(h5py.File(f, 'r')['/depth'])
        except Exception as ex:
            print(f"==> Data loading failure: {path}")
            raise ex

        assert data is not None
        return data

    def read_scannet_depth(self, path):
        if str(path).startswith('s3://'):
            depth = self.load_array_from_s3(str(path), None, cv2.IMREAD_UNCHANGED)
        else:
            depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        depth = depth / 1000
        depth = torch.from_numpy(depth).float()  # (h, w)
        return depth

    def __getitem__(self, idx):
        data_name = self.data_names[idx]
        scene_name, scene_sub_name, stem_name_0, stem_name_1 = data_name
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'
        # read the grayscale image which will be resized to (1, 480, 640)
        img_name0 = osp.join(self.root_dir, scene_name, 'color', f'{stem_name_0}.jpg')
        img_name1 = osp.join(self.root_dir, scene_name, 'color', f'{stem_name_1}.jpg')
        # TODO: Support augmentation & handle seeds for each worker correctly.
        image0 = cv2.imread(img_name0)[:, :, [2, 1, 0]]
        image0 = cv2.resize(image0, (640, 480))
                                #    augment_fn=np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        image1 = cv2.imread(img_name1)[:, :, [2, 1, 0]]
        image1 = cv2.resize(image1, (640, 480))
                                #    augment_fn=np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        # read the depthmap which is stored as (480, 640)
        depth0 = self.read_scannet_depth(osp.join(self.root_dir, scene_name, 'depth', f'{stem_name_0}.png'))
        depth1 = self.read_scannet_depth(osp.join(self.root_dir, scene_name, 'depth', f'{stem_name_1}.png'))
        # read the intrinsic of depthmap
        K = torch.tensor(self.intrinsics[scene_name].copy(), dtype=torch.float).reshape(3, 3)
        K = F.pad(K, [0, 1, 0, 1])
        # read and compute relative poses
        T_0to1 = K @ torch.tensor(self._compute_rel_pose(scene_name, stem_name_0, stem_name_1),
                              dtype=torch.float32) @ torch.inverse(K)
        # T_1to0 = T_0to1.inverse()
        return T_0to1, image0, image1, depth0, depth1, osp.join(scene_name, 'color', f'{stem_name_0}.jpg'), \
            osp.join(scene_name, 'color', f'{stem_name_1}.jpg')