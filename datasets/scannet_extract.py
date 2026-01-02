import torch
import torch.nn.functional as F
import numpy as np
import sys
import cv2
import h5py
import copy
import math
import glob
import os
from itertools import combinations
# from scipy.spatial.transform import Rotation as R
# sys.path.append(r"/home/junjieni/research/FDesp/")
from utils.utils import Resize_depth, Get_pairs, Get_params, Create_P_resize, Resize_img, Get_resize_ratio
import multiprocessing as mp
import torchvision.models as models
from models.first_layer import feature_extractor
from utils.coco_utils import compute_F
import collections

def get_intrinsic_from_txt(intrinsic_path):
    intrinsic_ = np.asarray(open(intrinsic_path).readlines())
    intrinsic = []
    for j in range(intrinsic_.shape[0]):
        intrinsic.append(intrinsic_[j].split())
        intrinsic[j][-1] = intrinsic[j][-1][:-1]
    intrinsic = np.asarray(intrinsic).astype(float)
    return intrinsic    

# data_path就是scanet
def create_scannet_pairs(data_path, save_path, set):
    if set == "train":
        set_file = open(data_path + "Tasks/Benchmark/scannetv1_train.txt", "r")
    else:
        set_file = open(data_path + "Tasks/Benchmark/scannetv1_test.txt", "r")
    scenes_name = np.sort(np.asarray(set_file.readlines()))
    pairs_file = open(save_path + "pairs_list_" + set + ".txt", "w")
    for i in range(scenes_name.shape[0]):
        img_path = data_path + "scans/" + scenes_name[i][:-1] + "/color/"
        # depth_path = data_path + scenes_name + "/depth/"
        figure_num = np.asarray(os.listdir(img_path)).shape[0]
        for j in range(figure_num - 50):
            scenes_name_now = scenes_name[i][:-1]
            label = str(j)
            pairs_file.write(scenes_name_now + " " + label + "\n")
    intrinsics = {}
    scenes_name = np.sort(np.asarray(os.listdir(data_path + "scans")))
    for i in range(scenes_name.shape[0]):
        if scenes_name[i][:5] == "scene":
            scenes_path = data_path + "scans/" + scenes_name[i] + "/intrinsic/"
            intrinsic_colar = get_intrinsic_from_txt(scenes_path + "intrinsic_color.txt")
            intrinsic_depth = get_intrinsic_from_txt(scenes_path + "intrinsic_depth.txt")
            intrinsics[scenes_name[i] + "_colar"] = intrinsic_colar
            intrinsics[scenes_name[i] + "_depth"] = intrinsic_depth
    np.save(save_path + "intrinsics", intrinsics)

def scannet_get_pairs(param_path, set):
    pair_file = np.asarray(open(param_path + "pairs_list_" + set + ".txt").readlines())
    pairs = []
    for i in range(pair_file.shape[0]):
        line = pair_file[i].split()
        pairs.append([line[0], line[1], str(int(line[1]) + 50)])
    return np.asarray(pairs)

def get_scannet_pose(pose_path):
    pose_ = np.asarray(open(pose_path).readlines())
    pose = []
    for j in range(pose_.shape[0]):
        pose.append(pose_[j].split())
        pose[j][-1] = pose[j][-1][:-1]
    pose = np.asarray(pose)
    return pose


def Compute_depth_label(depth0, depth1, P, num, patch_size, threthold=1):
    # patch_size is the distance between the point and the board of patch
    device = depth0.device
    upper_bound = 1e7
    lower_bound = 1e-11
    row_num0 = depth0.shape[1] // patch_size // 2
    col_num0 = depth0.shape[2] // patch_size // 2
    row_num1 = depth1.shape[1] // patch_size // 2
    col_num1 = depth1.shape[2] // patch_size // 2
    # 在2×2 的区间排除无数据区域进行平均，在全部没有数据的情况下权值则全部设定为1
    # row = torch.arange(0, col_num0, device=device) * 2 * patch_size + patch_size - 1
    # rows = row.reshape(1, col_num0).repeat(row_num0, 1).reshape(1, row_num0 * col_num0, 1).repeat(num, 1, 1)
    # col = torch.arange(0, row_num0, device=device) * 2 * patch_size + patch_size - 1
    # cols = col.reshape(row_num0, 1).repeat(1, col_num0).reshape(1, row_num0 * col_num0, 1).repeat(num, 1, 1)
    # prefix = torch.arange(0, num, device=device).reshape(num, 1).repeat(1, row_num0 * col_num0).reshape(num, row_num0*col_num0, 1).long()
    # rows_new = rows.long()
    # cols_new = cols.long()
    # depth_input_row = rows_new
    # depth_input_col = cols_new
    # d0 = depth0[prefix, depth_input_col, depth_input_row]
    # d0 = d0.reshape(num, col_num0 * row_num0, 1)

    row = torch.arange(0, col_num0, device=device) * 2 * patch_size + patch_size - 1
    rows = row.reshape(1, col_num0).repeat(row_num0, 1).reshape(1, row_num0 * col_num0, 1).repeat(num, 1, 1)
    col = torch.arange(0, row_num0, device=device) * 2 * patch_size + patch_size - 1
    cols = col.reshape(row_num0, 1).repeat(1, col_num0).reshape(1, row_num0 * col_num0, 1).repeat(num, 1, 1)
    prefix = torch.arange(0, num, device=device).reshape(num, 1).repeat(1, row_num0 * col_num0 * 4).reshape(num, row_num0*col_num0, 4)
    rows_new = rows.long()
    cols_new = cols.long()
    depth_input_row = torch.cat([rows_new, rows_new, rows_new + 1, rows_new + 1], dim=2)
    depth_input_col = torch.cat([cols_new, cols_new + 1, cols_new, cols_new + 1], dim=2)
    d0 = depth0[prefix, depth_input_col, depth_input_row]
    d0_weights = (d0 > lower_bound).int()
    d0_weights[d0.max(2)[0] < lower_bound] = 1
    d0 = d0.reshape(num, col_num0 * row_num0, 4)
    d0 = ((d0 * d0_weights).sum(2) / d0_weights.sum(2)).reshape(num, col_num0 * row_num0, 1)

    if_d0 = d0 < lower_bound
    d0[d0 < lower_bound] = upper_bound
    # 从左向右投影
    last_ones = torch.ones([num, col_num0*row_num0, 1], device=device)
    point_input = torch.cat([(rows + 1.0) * d0, (cols + 1.0) * d0, d0, last_ones], dim=2)
    point_output = torch.einsum('ijk,ipk->ipj', P, point_input)
    point_output[:, :, 0] = point_output[:, :, 0] / point_output[:, :, 2]
    point_output[:, :, 1] = point_output[:, :, 1] / point_output[:, :, 2]
    correspondense = torch.zeros([point_output.shape[0], point_output.shape[1], 3], device=device)
    correspondense[:, :, 0:2] = point_output[:, :, 0:2]
    # 去掉投出图像的投影
    if_outlier1 = torch.logical_and(torch.logical_or(point_output[:, :, 0] < 2, point_output[:, :, 0] >= depth1.shape[2] - 2), torch.logical_not(if_d0.reshape(d0.shape[0], d0.shape[1])))
    if_outlier2 = torch.logical_and(torch.logical_or(point_output[:, :, 1] < 2, point_output[:, :, 1] >= depth1.shape[1] - 2), torch.logical_not(if_d0.reshape(d0.shape[0], d0.shape[1])))
    correspondense[if_d0.repeat(1, 1, 3)] = -upper_bound
    # 计算右图目标点的深度
    output_row = torch.round(point_output[:, :, 0] - 0.5).reshape(num, row_num0 * col_num0, 1).int()
    output_row[torch.logical_or(point_output[:, :, 0] < 2, point_output[:, :, 0] >= depth1.shape[2] - 2)] = int(depth1.shape[2] / 2)
    output_col = torch.round(point_output[:, :, 1] - 0.5).reshape(num, row_num0 * col_num0, 1).int()
    output_col[torch.logical_or(point_output[:, :, 1] < 2, point_output[:, :, 1] >= depth1.shape[1] - 2)] = int(depth1.shape[1] / 2)
    output_cols = output_col.repeat(1, 1, 3)
    depth1_input_row = torch.cat((output_row - 1, output_row, output_row + 1), dim=2).repeat(1, 1, 3).long()
    depth1_input_col = torch.cat((output_cols - 1, output_cols, output_cols + 1), dim=2).long()
    d1_weights = torch.clamp(1 - (- depth1_input_row + point_output[:, :, 0, None] - 0.5).abs(), min=0.0, max=1.0) * \
        torch.clamp(1 - (- depth1_input_col + point_output[:, :, 1, None] - 0.5).abs(), min=0.0, max=1.0)
    # d1_weights = torch.clamp(1 - depth1_input_row + point_output[:, :, 0, None] - 0.5, min=0.0, max=1.0) * \
    #     torch.clamp(1 - depth1_input_col + point_output[:, :, 1, None] - 0.5, min=0.0, max=1.0)
    prefix = torch.arange(num, device=device).reshape(num, 1).repeat(1, row_num0 * col_num0 * 9).reshape(num, row_num0 * col_num0, 9).long()
    d1 = depth1[prefix, depth1_input_col, depth1_input_row]
    d1_weights = (d1 > lower_bound).float() * d1_weights
    d1_weights[torch.max(d1, dim=2)[0] < lower_bound] = 1
    # d1_weights[d1 - np.min(d1, axis=2)[0].reshape(d1.shape[0], d1.shape[1], 1).repeat(4, axis=2) > 1] = 0
    d1 = ((d1 * d1_weights).sum(2) / d1_weights.sum(2)).reshape(num, col_num0 * row_num0, 1)
    d1[d1 < lower_bound] = upper_bound
    # 重投影，右图向左图
    point_input2 = torch.cat([output_row * d1, output_col * d1, d1, last_ones], dim=2)
    point_output2 = torch.einsum('ijk,ipk->ipj', torch.linalg.inv(P), point_input2)
    point_output2[:, :, 1] = point_output2[:, :, 1] / point_output2[:, :, 2]
    point_output2[:, :, 0] = point_output2[:, :, 0] / point_output2[:, :, 2]
    # 计算重投影误差
    distance_vector = torch.abs(point_input / d0 - point_output2)
    distance = torch.sqrt(torch.pow(distance_vector[:, :, 0], 2) + torch.pow(distance_vector[:, :, 1], 2))
    # print(distance)
    # correspondense[:, :, 2] = distance
    correspondense[:, :, 2] = d0[:, :, 0] / d1[:, :, 0]
    correspondense[distance.reshape(distance.shape[0], distance.shape[1], 1).repeat(1, 1, 3) > threthold] = -upper_bound
    correspondense[:, :, :2][if_outlier1] = -upper_bound
    correspondense[:, :, :2][if_outlier2] = -upper_bound
    return correspondense


def create_scannet_label(K, left_depth, right_depth, lp, rp):
    kd = K[0:3, 0:3]
    P = K.dot(rp).dot(np.linalg.inv(K.dot(lp)))
    label = Compute_depth_label(left_depth.reshape(1, left_depth.shape[0], left_depth.shape[1]), right_depth.reshape(1, right_depth.shape[0], right_depth.shape[1]), P.reshape(1, 4, 4), 1, 16)
    label_reverse = Compute_depth_label(right_depth.reshape(1, right_depth.shape[0], right_depth.shape[1]), left_depth.reshape(1, left_depth.shape[0], left_depth.shape[1]), np.linalg.inv(P).reshape(1, 4, 4), 1, 16)
    pose = rp.dot(np.linalg.inv(lp))
    transform = np.array([[0, -pose[2, 3], pose[1, 3]], [pose[2, 3], 0, -pose[0, 3]], [-pose[1, 3], pose[0, 3], 0]])
    E = transform.dot(pose[:3, :3])
    F = np.linalg.inv(kd).transpose().dot(E).dot(np.linalg.inv(kd))
    label = np.concatenate([label[0], F], axis=0)
    label_reverse = np.concatenate([label_reverse[0], F.transpose(1, 0)], axis=0)
    return label, label_reverse


def create_megadepth_label(P, left_depth, right_depth):
    label = Compute_depth_label(left_depth, right_depth, P, P.shape[0], 1)
    label_reverse = Compute_depth_label(right_depth, left_depth, torch.linalg.inv(P), P.shape[0], 1)
    return label, label_reverse

# def create_megadepth_label_refine(left_K, right_K, lp, rp):
#     pose = rp.dot(np.linalg.inv(lp))
#     transform = np.array([[0, -pose[2, 3], pose[1, 3]], [pose[2, 3], 0, -pose[0, 3]], [-pose[1, 3], pose[0, 3], 0]])
#     E = transform.dot(pose[:3, :3])
#     F = np.linalg.inv(right_K[:3, :3]).transpose().dot(E).dot(np.linalg.inv(left_K[:3, :3]))
#     label = F
#     label_reverse = F.transpose(1, 0)
#     return label, label_reverse

def create_megadepth_label_refine(left_K, right_K, left_depth, right_depth, lp, rp):
    P = right_K.dot(rp).dot(np.linalg.inv(left_K.dot(lp)))
    label = Compute_depth_label(left_depth.reshape(1, left_depth.shape[0], left_depth.shape[1]), right_depth.reshape(1, right_depth.shape[0], right_depth.shape[1]), P.reshape(1, 4, 4), 1, 4, threthold=4.0)
    label_third = Compute_depth_label(left_depth.reshape(1, left_depth.shape[0], left_depth.shape[1]), right_depth.reshape(1, right_depth.shape[0], right_depth.shape[1]), P.reshape(1, 4, 4), 1, 1, threthold=1.0)
    # label_reverse = Compute_depth_label(right_depth.reshape(1, right_depth.shape[0], right_depth.shape[1]), left_depth.reshape(1, left_depth.shape[0], left_depth.shape[1]), np.linalg.inv(P).reshape(1, 4, 4), 1, 4, threthold=2)
    pose = rp.dot(np.linalg.inv(lp))
    transform = np.array([[0, -pose[2, 3], pose[1, 3]], [pose[2, 3], 0, -pose[0, 3]], [-pose[1, 3], pose[0, 3], 0]])
    E = transform.dot(pose[:3, :3])
    F = np.linalg.inv(right_K[:3, :3]).transpose().dot(E).dot(np.linalg.inv(left_K[:3, :3]))
    label = np.concatenate([label[0], label_third[0], F], axis=0)
    label_reverse = F.transpose(1, 0)
    return label, label_reverse

def create_scannet_label_refine(K, left_depth, right_depth, lp, rp):
    kd = K[0:3, 0:3]
    P = K.dot(rp).dot(np.linalg.inv(K.dot(lp)))
    label = Compute_depth_label(left_depth.reshape(1, left_depth.shape[0], left_depth.shape[1]), right_depth.reshape(1, right_depth.shape[0], right_depth.shape[1]), P.reshape(1, 4, 4), 1, 4, threthold=4.0)
    label_third = Compute_depth_label(left_depth.reshape(1, left_depth.shape[0], left_depth.shape[1]), right_depth.reshape(1, right_depth.shape[0], right_depth.shape[1]), P.reshape(1, 4, 4), 1, 1, threthold=4.0)
    pose = rp.dot(np.linalg.inv(lp))
    transform = np.array([[0, -pose[2, 3], pose[1, 3]], [pose[2, 3], 0, -pose[0, 3]], [-pose[1, 3], pose[0, 3], 0]])
    E = transform.dot(pose[:3, :3])
    F = np.linalg.inv(kd).transpose().dot(E).dot(np.linalg.inv(kd))
    label = np.concatenate([label[0], label_third[0], F], axis=0)
    label_reverse = F.transpose(1, 0)
    return label, label_reverse

def create_scannet_label_refine2(K, lp, rp):
    kd = K[0:3, 0:3]
    pose = rp.dot(np.linalg.inv(lp))
    transform = np.array([[0, -pose[2, 3], pose[1, 3]], [pose[2, 3], 0, -pose[0, 3]], [-pose[1, 3], pose[0, 3], 0]])
    E = transform.dot(pose[:3, :3])
    F = np.linalg.inv(kd).transpose().dot(E).dot(np.linalg.inv(kd))
    label = F
    label_reverse = F.transpose(1, 0)
    return label, label_reverse

# data_path = "/mnt/nas_7/datasets/ScanNet/"
# save_path = "/mnt/nas_8/group/nijunjie/Scannet_parameters/"
# create_scannet_pairs(data_path, save_path, "test")
