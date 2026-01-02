from numpy.core.fromnumeric import shape
import torch
import torch.nn.functional as F
import numpy as np
import sys
import cv2
import os
from utils.utils import Create_K_resize, origin_extract

def Get_coco_pairs(path):
    path_1 = path + "train2017/"
    path_2 = path + "train2017_transform1/"
    path_3 = path + "train2017_transform2/"
    path_4 = path + "train2017_transform3/"
    path_5 = path + "train2017_transform4/"
    path_6 = path + "train2017_transform5/"
    photo_name = np.sort(np.array(os.listdir(path_1)))
    pairs = []
    for i in range(photo_name.shape[0]):
        pairs.append([path_1 + photo_name[i], path_2 + photo_name[i]])
        pairs.append([path_1 + photo_name[i], path_3 + photo_name[i]])
        pairs.append([path_1 + photo_name[i], path_4 + photo_name[i]])
        pairs.append([path_1 + photo_name[i], path_5 + photo_name[i]])
        pairs.append([path_1 + photo_name[i], path_6 + photo_name[i]])
    return np.array(pairs)

# 这里假设input的结构为：batch_size*匹配对的数量 ×（左图patch顺位x, 左图patch顺位y, 中心匹配估计x，
# 中心匹配估计y， 96×96单元格的右scalex， 96×96单元格的右scaley， 96×96单元格的左scalex， 96×96单元格的左scaley）
# 此处的scale指的是覆盖范围相比标准值的比例
# H已经进行过取逆的变换, 并且已经转换为定义域为[0, 480],[0, 640]的像素坐标系，结构则是batch_size*匹配对的数量 × 3 × 3
# 输出的H_new, 原点都是96×96的patch的左上角位置
def Compute_resized_homography(H, input, width = 640, height = 480, patch_size_high = 32):
    input[:, 2] = input[:, 2]
    input[:, 3] = input[:, 3]
    input[:, 0] = input[:, 0] + 0.5
    input[:, 1] = input[:, 1] + 0.5
    M1 = torch.eye(3, device=H.device).reshape(1, 3, 3).repeat(input.shape[0], 1, 1)
    M2 = torch.eye(3, device=H.device).reshape(1, 3, 3).repeat(input.shape[0], 1, 1)
    M1[:, 0:2, 2] = - (input[:, 2:4] / input[:, 4:6] - 48)
    M1[:, 0, 0] = 1 / input[:, 4]
    M1[:, 1, 1] = 1 / input[:, 5]
    M2[:, 0, 0] = input[:, 6]
    M2[:, 1, 1] = input[:, 7]
    M2[:, 0:2, 2] = (input[:, 0:2] * patch_size_high) - input[:, 6:8] * 48
    H_new = torch.einsum('iqj,ijk,ikp->iqp', M1.double(), H, M2.double())
    H_new_reverse = torch.inverse(H_new)
    # test = torch.tensor([[368.0], [272.0], [1.0]], device=H.device).double()
    # test2 = torch.tensor([[48.0], [48.0], [1.0]], device=H.device).double()
    # test2 = torch.mm(H_new[55].double(), test2)
    # test = torch.mm(H[0].double(), test)
    # print(H[0], input[55], H_new[55], test / test[2], test2 / test2[2])
    return H_new, H_new_reverse

def Compute_resized_epipolar(F, input, patch_size_high = 32):
    input[:, 2] = input[:, 2]
    input[:, 3] = input[:, 3]
    input[:, 0] = input[:, 0] + 0.5
    input[:, 1] = input[:, 1] + 0.5
    M1 = torch.eye(3, device=F.device).reshape(1, 3, 3).repeat(input.shape[0], 1, 1)
    M2 = torch.eye(3, device=F.device).reshape(1, 3, 3).repeat(input.shape[0], 1, 1)
    M1[:, 0:2, 2] = - (input[:, 2:4] / input[:, 4:6] - 1.5 * patch_size_high)
    M1[:, 0, 0] = 1 / input[:, 4]
    M1[:, 1, 1] = 1 / input[:, 5]
    M1 = M1.inverse()
    M1 = M1.permute(0, 2, 1)
    M2[:, 0, 0] = input[:, 6]
    M2[:, 1, 1] = input[:, 7]
    M2[:, 0:2, 2] = (input[:, 0:2] * patch_size_high) - input[:, 6:8] * 1.5 * patch_size_high
    F_new = torch.einsum('iqj,ijk,ikp->iqp', M1.double(), F, M2.double())
    F_new = (F_new.permute(1, 2, 0) / (F_new[:, 2, 2] + 1e-10)).permute(2, 0, 1)
    F_new_reverse = F_new.permute(0, 2, 1)
    return F_new, F_new_reverse

import copy

def Compute_resized_refine_dense(label, input, patch_size_high = 32):
    label_use = copy.deepcopy(label)
    label_use[:, :, 2] = 1
    input[:, 2] = input[:, 2]
    input[:, 3] = input[:, 3]
    input[:, 0] = input[:, 0] + 0.5
    input[:, 1] = input[:, 1] + 0.5
    M1 = torch.eye(3, device=label.device).reshape(1, 3, 3).repeat(input.shape[0], 1, 1)
    M1[:, 0:2, 2] = - (input[:, 2:4] / input[:, 4:6] - 1.5 * patch_size_high)
    M1[:, 0, 0] = 1 / input[:, 4]
    M1[:, 1, 1] = 1 / input[:, 5]
    label_new = torch.einsum('iqj,iabj->iabq', M1.double(), label_use)
    criterion1 = torch.logical_and(label_new[:, :, :, 0] >= 8, label_new[:, :, :, 0] <= 88)
    criterion2 = torch.logical_and(label_new[:, :, :, 1] >= 8, label_new[:, :, :, 1] <= 88)
    criterion = torch.logical_and(criterion1, criterion2)
    label_new[:, :, :, 0] = torch.where(criterion, label_new[:, :, :, 0], torch.tensor(-1.0, device=label.device).double())
    criterion1 = torch.logical_and(label_new[:, :, :, 0] >= 0, label_new[:, :, :, 0] <= 96)
    criterion2 = torch.logical_and(label_new[:, :, :, 1] >= 0, label_new[:, :, :, 1] <= 96)
    criterion = torch.logical_and(criterion1, criterion2)
    label_new[:, :, :, 0] = torch.where(criterion, label_new[:, :, :, 0], torch.tensor(-100000.0, device=label.device).double())
    label_new[:, :, :, 0] = torch.where(torch.logical_and(label[:, :, :, 0] > -1.01, label[:, :, :, 0] < -0.99), torch.tensor(-10.0, device=label.device).double(), label_new[:, :, :, 0].double())
    return label_new


def compute_F(gt_now, gt_forward, K0, K1, size0, size1):
    K0_new, K1_new = Create_K_resize(K0, K1, size0, size1)
    K1 = K1_new[0:3, 0:3]
    gt_now = np.append(gt_now, np.array([0, 0, 0, 1]))
    gt_now = np.reshape(gt_now, (4, 4))
    # gt_now = np.linalg.inv(gt_now)
    t_now = -np.linalg.inv(gt_now[0:3, 0:3]).dot(gt_now[0:3, 3])
    gt_forward = np.append(gt_forward, np.array([0, 0, 0, 1]))
    gt_forward = np.reshape(gt_forward, (4, 4))
    # gt_forward = np.linalg.inv(gt_forward)
    t_forward = -np.linalg.inv(gt_forward[0:3, 0:3]).dot(gt_forward[0:3, 3])
    location_one_on_two = np.ones(4)
    location_one_on_two[0:3] = K1.dot(gt_forward[0:3, 0:3].dot(t_now - t_forward))
    # print(location_one_on_two)
    main_point2 = location_one_on_two[0:3]
    main_point2 = main_point2 / (main_point2[2] + 1e-10)
    # print(main_point2)
    H_2to1 = np.dot(K1_new.dot(gt_forward), np.linalg.pinv(K0_new.dot(gt_now)))
    # print(H_2to1)
    H_2to1 = H_2to1[0:3, 0:3]
    e2 = np.array([[0, -main_point2[2], main_point2[1]],
                   [main_point2[2], 0, -main_point2[0]],
                   [-main_point2[1], main_point2[0], 0]])
    F = np.dot(e2, H_2to1).transpose()
    F = F / (F[2, 2] + 1e-10)
    return F

#最终得到的label实际上是(b, 12*12+9, 1),其中label[:, :12*12, 0]指示的是has matches,unknown, doesn't have matches
def eval_epipolar_label(F, patch_row=96, patch_col=96, patch_size=8, if_shrink=False):
    if if_shrink:
        patch_row = patch_row * 2 // 3
        patch_col = patch_col * 2 // 3
    point_col = patch_col // patch_size
    point_row = patch_row // patch_size
    point_input_x = torch.arange(patch_size // 2, patch_col, patch_size, device=F.device).reshape(1, 1, point_col, 1).repeat(F.shape[0], point_row, 1, 1)
    point_input_y = torch.arange(patch_size // 2, patch_row, patch_size, device=F.device).reshape(1, point_row, 1, 1).repeat(F.shape[0], 1, point_col, 1)
    point_input_z = torch.ones([F.shape[0], point_row, point_col, 1], device=F.device)
    point_input = torch.cat([point_input_x, point_input_y, point_input_z], dim=3)
    point_input = point_input.reshape(F.shape[0], point_row*point_col, 3).double()
    centre_point = torch.tensor([patch_row/2, patch_col/2, 1], device=F.device).\
        reshape(1, 1, 3).repeat(F.shape[0], point_row*point_col, 1).double()
    if if_shrink:
        point_input[:, :, 0:2] += 16
        centre_point[:, :, 0:2] += 16
    label0 = torch.einsum('itj,ijk,itk->it', centre_point, F, point_input).abs()
    line1 = torch.einsum('ijk,itk->itj', F, point_input).abs()
    label0 = label0/(torch.norm(line1[:, :, :2], dim=2, keepdim=False) + 1e-8)
    label = torch.where(label0 < patch_row // 2 - 8, 100000.0, -10.0).double()
    label = torch.where(label0 >= patch_row // 2, -100000.0, label)
    label = torch.cat([label, F.reshape(F.shape[0], 9)], dim=1).unsqueeze(2)
    return label

def Compute_dense_label_H(label, reverse_label):
    point_input_origin_x = torch.arange(16, 656, 32, device=label.device).reshape(1, 1, 20, 1).repeat(label.shape[0],
                                                                                                     15, 1, 1)
    point_input_origin_y = torch.arange(16, 496, 32, device=label.device).reshape(1, 15, 1, 1).repeat(label.shape[0],
                                                                                                     1, 20, 1)
    point_input_origin_z = torch.ones([label.shape[0], 15, 20, 1], device=label.device)
    point_input_origin = torch.cat([point_input_origin_x, point_input_origin_y, point_input_origin_z], dim=3)
    label_used = torch.einsum('ipl,ijkl->ijkp', label, point_input_origin.double()).reshape(-1, 300, 3)
    # print(label_used[0, 51], point_input_origin.reshape(300, 3)[51])
    label_used = (label_used.permute(2, 0, 1) / label_used[:, :, 2]).permute(1, 2, 0)
    # img_show(left[0].cpu().numpy(), right[0].cpu().numpy(), label_used[0].cpu().numpy(), 1000)
    label_used_reverse = torch.einsum('ipl,ijkl->ijkp', reverse_label, point_input_origin.double()).reshape(-1, 300, 3)
    label_used_reverse = (label_used_reverse.permute(2, 0, 1) / label_used_reverse[:, :, 2]).permute(1, 2, 0)
    criterion1 = torch.logical_or(
        torch.logical_or(label_used[:, :, 0] < 0, label_used[:, :, 0] >= 640),
        torch.logical_or(label_used[:, :, 1] < 0, label_used[:, :, 1] >= 480))
    criterion2 = torch.logical_or(
        torch.logical_or(label_used_reverse[:, :, 0] < 0, label_used_reverse[:, :, 0] >= 640),
        torch.logical_or(label_used_reverse[:, :, 1] < 0, label_used_reverse[:, :, 1] >= 480))
    label_used = torch.where(criterion1, torch.tensor(-1000.0, device=label.device).float(),
                                    label_used.permute(2, 0, 1).float()).permute(1, 2, 0)
    label_used_reverse = torch.where(criterion2, torch.tensor(-1000.0, device=label.device).float(),
                                            label_used_reverse.permute(2, 0, 1).float()).permute(1, 2, 0)
    return label_used, label_used_reverse

def Compute_dense_refine_label_H(label, refine_input, if_nomatching1):
    label_refine, label_refine_reverse = Compute_resized_homography(
        label.reshape(label.shape[0], 1, 3, 3).repeat(1, 300, 1, 1)[
            torch.logical_not(if_nomatching1)], refine_input)
    # 计算对应点的位置
    point_input_x = torch.arange(4, 100, 8, device=label.device).reshape(1, 1, 12, 1).repeat(refine_input.shape[0], 12,
                                                                                            1, 1)
    point_input_y = torch.arange(4, 100, 8, device=label.device).reshape(1, 12, 1, 1).repeat(refine_input.shape[0], 1,
                                                                                            12, 1)
    point_input_z = torch.ones([refine_input.shape[0], 12, 12, 1], device=label.device)
    point_input = torch.cat([point_input_x, point_input_y, point_input_z], dim=3)
    label_refine_used = torch.einsum('ipl,ijkl->ijkp', label_refine, point_input.double()).reshape(-1, 144, 3)
    label_refine_used = (label_refine_used.permute(2, 0, 1) / label_refine_used[:, :, 2]).permute(1, 2, 0)
    label_refine_used_reverse = torch.einsum('ipl,ijkl->ijkp', label_refine_reverse, point_input.double()) \
        .reshape(-1, 144, 3)
    label_refine_used_reverse = (
            label_refine_used_reverse.permute(2, 0, 1) / label_refine_used_reverse[:, :, 2]).permute(1, 2, 0)
    criterion1 = torch.logical_or(
        torch.logical_or(label_refine_used[:, :, 0] <= 0, label_refine_used[:, :, 0] >= 96),
        torch.logical_or(label_refine_used[:, :, 1] <= 0, label_refine_used[:, :, 1] >= 96))
    criterion2 = torch.logical_or(
        torch.logical_or(label_refine_used_reverse[:, :, 0] <= 0, label_refine_used_reverse[:, :, 0] >= 96),
        torch.logical_or(label_refine_used_reverse[:, :, 1] <= 0, label_refine_used_reverse[:, :, 1] >= 96))
    label_refine_used = torch.where(criterion1, torch.tensor(-1000.0, device=label.device).float(),
                                    label_refine_used.permute(2, 0, 1).float()).permute(1, 2, 0)
    label_refine_used_reverse = torch.where(criterion2, torch.tensor(-1000.0, device=label.device).float(),
                                            label_refine_used_reverse.permute(2, 0, 1).float()).permute(1, 2, 0)
    return label_refine_used, label_refine_used_reverse

# 这里的average_point是（y, x, 1）的顺序
def Extract_refine(left, right, average_point, if_nomatching, patch_scale_high=8, patch_scale_low=2, width=96, height=96):
    # 本来应该- patch_scale_high // 2 × 3,但是因为要考虑pading,就要少减一个
    average_point = torch.round(average_point * patch_scale_high) - patch_scale_high // 2
    average_point_new = torch.zeros([average_point.shape[0], average_point.shape[1], average_point.shape[2]], device=left.device)
    average_point_new[:, :, 0] = average_point[:, :, 1]
    average_point_new[:, :, 1] = average_point[:, :, 0]
    #extract以前先把图像顺序正过来
    right = F.pad(right.permute(0, 3, 1, 2), (patch_scale_high, patch_scale_high, patch_scale_high, patch_scale_high), "constant", 0)
    patch_scale = 3 * patch_scale_high
    sequence_base_x = torch.arange(0, patch_scale, device=left.device).reshape(1, 1, 1, patch_scale, 1).\
        repeat(left.shape[0], 144, patch_scale, 1, 1)
    sequence_base_y = torch.arange(0, patch_scale, device=left.device).reshape(1, 1, patch_scale, 1, 1).\
        repeat(left.shape[0], 144, 1, patch_scale, 1)
    sequence_base = torch.cat([sequence_base_x, sequence_base_y], dim=4)
    sequence_base = (sequence_base.permute(2, 3, 0, 1, 4) + average_point_new).permute(2, 3, 0, 1, 4)
    sequence_index = sequence_base[:, :, :, :, 1] * (width + 2 * patch_scale_high) + sequence_base[:, :, :, :, 0]
    new_right = torch.gather(right.reshape(right.shape[0], 3, -1), index=sequence_index.reshape(right.shape[0], 1, -1).repeat(1, 3, 1).long(), dim=2).\
        reshape(-1, 3, 144, patch_scale, patch_scale).permute(0, 2, 1, 3, 4)[torch.logical_not(if_nomatching)]
    left_use = F.pad(left.permute(0, 3, 1, 2), (patch_scale_high, patch_scale_high, patch_scale_high, patch_scale_high), "constant", 0)
    new_left = origin_extract(left_use, patch_scale_high, width // patch_scale_high, height // patch_scale_high).permute(0, 2, 1, 3, 4)[torch.logical_not(if_nomatching)]
    return new_left, new_right, average_point_new + patch_scale_high // 2

def Compute_accuracy_indicator(average_error, label, classify_accurate, choice_rate, if_second=False):
    zero = label.new_tensor(0.0, device=label.device)
    x1 = classify_accurate
    if if_second:
        average_error = average_error.reshape(1, -1)
        x2 = (label[:, 0] > -0.01).reshape(1, -1)
        median = torch.median(average_error)
    else:
        average_error = average_error.reshape(label.shape[0], label.shape[1])
        x2 = (label[:, :, 0] > -0.01)
        median = torch.zeros(label.shape[0], device=label.device) + 10
        for j in range(label.shape[0]):
            if x2[j].float().sum() > 0.5:
                median[j] = average_error[j][x2[j]].median()
    mean = torch.mean(average_error[x2])
    x3 = torch.logical_and(x2, torch.logical_and(average_error < 2 * mean, average_error < 32))
    num1 = x1.float().sum()
    num2 = x2.float().sum()
    num3 = x3.float().sum()
    # print(torch.logical_not(x2).float().sum())
    # print((seg==9).float().sum())
    # accuracy1 = torch.logical_and(true_choice.reshape(-1), x2).float().sum() / (num2 + 1e-7)
    accuracy1 = num1 / (x1.shape[0])
    accuracy2 = choice_rate
    position_error = (torch.where(x3, average_error, torch.zeros_like(average_error)).float().sum(1) / (x3.float().sum(1) + 1e-9)).mean()
    # print((torch.where(x3, average_error, torch.zeros_like(average_error)).float().sum(1) / (x3.float().sum(1) + 1e-9)).mean())
    accuracy3 = (torch.where(x2, (average_error < 5.0).float(), torch.zeros_like(average_error)).float().sum(1) / (x2.float().sum(1) + 1e-9)).mean()
    # accuracy3 = torch.sum(torch.where(x2, (average_error.reshape(-1) - average_error_center.reshape(-1)).float() / (average_error.reshape(-1) + 1e-7), 
    #     torch.zeros_like(average_error.reshape(-1))).float()) / (num2 + 1e-7).float()
    return accuracy1, accuracy2, accuracy3, position_error
