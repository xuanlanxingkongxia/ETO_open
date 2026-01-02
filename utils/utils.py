import torch
import torch.nn.functional as F
import numpy as np
import cv2
import copy
import os
import tqdm

circle_num = 0


def Get_cameras(path, img_folder, use_set='train', if_origin=False):
    if use_set == "train":
        set_file = path + "megadepth_train_scenes.txt"
    if use_set == "test":
        set_file = path + "megadepth_validation_scenes_full.txt"
    f = open(set_file, 'r')
    lines = np.array(f.readlines())
    images = {}
    Image = {}
    for i in range(lines.shape[0]):
        if not (os.path.exists(path + lines[i][:-1])):
            continue
        dense_name = np.sort(np.array(os.listdir(path + lines[i][:-1])))
        for j in range(dense_name.shape[0]):
            path_now = path + lines[i][:-1] + "/" + dense_name[j]
            img_cam_txt_path = os.path.join(path_now, 'img_cam.txt')
            img_cam_txt_path2 = os.path.join(path_now, 'img_cam_new.txt')
            with open(img_cam_txt_path, "r") as fid:
                fid2 = open(img_cam_txt_path2, "r") 
                while True:
                    line = fid.readline()
                    line2 = fid2.readline()
                    if not line or not line2:
                        break
                    line = line.strip()
                    line2 = line2.strip()
                    if len(line) > 0 and line[0] != "#":
                        elems = line.split()
                        elems2 = line2.split()
                        image_name = elems[0]
                        img_path = os.path.join(img_folder + "img/" + lines[i][:-1] + "/" + dense_name[j] , image_name) + ".npy"
                        w, h = int(elems2[1]), int(elems2[2])
                        fx, fy = float(elems2[3]), float(elems2[4])
                        cx, cy = float(elems2[5]), float(elems2[6])
                        P = np.array(elems[7:19]).reshape(3, 4)
                        last_line = np.array([0, 0, 0, 1])
                        P = np.concatenate([P, last_line.reshape(1, -1)], axis=0)
                        R = np.array(P[0:3, 0:3])
                        T = np.array(P[0:3, 3])
                        intrinsic = np.array([[fx, 0, cx],
                                              [0, fy, cy],
                                              [0, 0, 1]])
                        if if_origin==False:
                            r1, add_num1 = Get_resize_ratio(np.array([w, h]), np.array([640, 480]))
                            k1 = np.identity(4)
                            k1[0:3, 0:3] = r1 * intrinsic[0:3, 0:3]
                            k1[2, 2] = 1
                            k1[0:2, 2] -= add_num1 * r1
                            images[img_path] = {
                                "name":image_name, "K":k1, "R":R, "T":T, "P":P}
                        else:
                            # img_path2 = img_folder + lines[i][:-1] + "/" + dense_name[j] + "/imgs/" + image_name
                            max_shape = max(w, h)
                            size = 1600.0 / max_shape
                            r1, add_num1 = Get_resize_ratio(np.array([w, h]), np.array([int(w * size), int(h *  size)]))
                            # r1, add_num1 = Get_resize_ratio(np.array([w, h]), np.array([640, 480]))
                            k1 = np.identity(4)
                            k1[0:3, 0:3] = r1 * intrinsic[0:3, 0:3]
                            k1[2, 2] = 1
                            k1[0:2, 2] -= add_num1 * r1
                            images[img_path] = {
                                "name":image_name, "K":k1, "R":R, "T":T, "P":P, "h": int(h*size), "w": int(w*size)}
    return images


def get_pose_error(kpts0, kpts1, K0, K1, pose, imf1, imf2, threshold, num):
    transform = np.array([[0, -pose[2, 3], pose[1, 3]], [pose[2, 3], 0, -pose[0, 3]], [-pose[1, 3], pose[0, 3], 0]])
    E_gt = transform.dot(pose[:3, :3])
    F_gt = np.linalg.inv(K1).transpose().dot(E_gt).dot(np.linalg.inv(K0))
    third_line = np.ones([kpts0.shape[0], 1])
    p1 = np.concatenate([kpts0, third_line], axis=1)
    p2 = np.concatenate([kpts1, third_line], axis=1)
    line = np.einsum("jk,ik->ij", F_gt, p1)
    result = np.abs(np.einsum("ij,jk,ik->i", p2, F_gt, p1)) / np.sqrt(line[:, 0] ** 2 + line[:, 1] ** 2)
    # result_masked = np.einsum("ij,jk,ik->i", p2[mask2], F_gt, p1[mask2]) / np.sqrt(line[mask2][:, 0] ** 2 + line[mask2][:, 1] ** 2)

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = threshold / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=1-1e-5,
        method=cv2.RANSAC)

    assert E is not None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    R_gt = pose[:3, :3]
    t_gt = pose[:3, 3]
    if ret != None:
        R, t, _ = ret
    try:
        error_t = angle_error_vec(t, t_gt)
        error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
        error_R = angle_error_mat(R, R_gt)
    except:
        error_t = np.array(1000.0)
        error_R = np.array(1000.0)

    result_mask = result[mask[:, 0].astype(float) > 0.5].mean()
    if np.isnan(result_mask).any() or np.isinf(result_mask).any():
        result_mask = np.array(3.0)
    return error_R, error_t, kpts0, kpts1, result, result_mask




def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))

def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))

def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.round(np.trapz(r, x=e)/t, 3))
    return aucs

def Compute_accuracy(images, results, scale_factor=1.0, threshold=0.32):
    R_errors = []
    T_errors = []
    exist_rates_32 = []
    exist_rates_8 = []
    recall_rates_1 = []
    recall_rates_2 = []
    recall_rates_3 = []
    recall_rates_4 = []
    recall_rates_5 = []
    recall_rates_6 = []
    recall_rates_sum_32 = []
    recall_rates_sum_8 = []
    accuracy_rates_1 = []
    accuracy_rates_2 = []
    accuracy_rates_3 = []
    accuracy_rates_sum = []
    point_errs = []
    point_errs_masked = []
    for i in tqdm.tqdm(range(results.__len__()), ncols=50):
        pair = results[i]
        if pair == -1:
            R_errors.append(1000)
            T_errors.append(1000)
            exist_rates_32.append(0)
            exist_rates_8.append(0)
            recall_rates_1.append(0)
            recall_rates_2.append(0)
            recall_rates_3.append(0)
            recall_rates_4.append(0)
            recall_rates_5.append(0)
            recall_rates_6.append(0)
            accuracy_rates_1.append(0)
            accuracy_rates_2.append(0)
            accuracy_rates_3.append(0)
            recall_rates_sum_32.append(1e-10)
            recall_rates_sum_8.append(1e-10)
            accuracy_rates_sum.append(1e-10)
            continue
        kp1 = pair["matches_l"].cpu().numpy()
        kp2 = pair["matches_r"].cpu().numpy()
        if kp1.shape[0] < 15:
            R_errors.append(1000)
            T_errors.append(1000)
            exist_rates_32.append(0)
            exist_rates_8.append(0)
            recall_rates_1.append(0)
            recall_rates_2.append(0)
            recall_rates_3.append(0)
            recall_rates_4.append(0)
            recall_rates_5.append(0)
            recall_rates_6.append(0)
            accuracy_rates_1.append(0)
            accuracy_rates_2.append(0)
            accuracy_rates_3.append(0)
            recall_rates_sum_32.append(1e-10)
            recall_rates_sum_8.append(1e-10)
            accuracy_rates_sum.append(1e-10)
            continue
        kp1 = np.concatenate([kp1[:, 0].reshape(-1, 1), kp1[:, 1].reshape(-1, 1)], axis=1)
        kp2 = np.concatenate([kp2[:, 0].reshape(-1, 1), kp2[:, 1].reshape(-1, 1)], axis=1)
        img1, img2 = images[pair["left_path"][0]], images[pair["right_path"][0]]
        intrinsic1, intrinsic2 = copy.copy(img1['K']), copy.copy(img2['K'])
        intrinsic2[:3, :3] = scale_intrinsics(intrinsic2[:3, :3], [1.0/ scale_factor, 1.0/ scale_factor])
        if scale_factor > 1.0:
            intrinsic1[:2, 2] += np.asarray([int((scale_factor-1)*320), int((scale_factor-1)*240)])
        else:
            intrinsic2[:2, 2] += np.asarray([int((1 - scale_factor)*320), int((1 - scale_factor)*240)])
        extrinsic1, extrinsic2 = img1['P'], img2['P']
        pose = extrinsic2.astype(float).dot(np.linalg.inv(extrinsic1.astype(float)))
        R_error, T_error, kp1, kp2, result_ori, result_masked = get_pose_error(kp1, kp2, intrinsic1[0:3, 0:3],
                intrinsic2[0:3, 0:3], pose, pair["left_path"][0], pair["right_path"][0], threshold=threshold, num=i)
        point_errs.append(np.mean(np.abs(result_ori)))
        point_errs_masked.append(np.mean(np.abs(result_masked)))
        results[i]["R_error"] = R_error
        results[i]["T_error"] = T_error
        results[i]["epipolar_distance"] = result_ori
        R_errors.append(max(R_error, T_error))
        T_errors.append(min(R_error, T_error))
    R_errors = np.array(R_errors)
    T_errors = np.array(T_errors)
    R_2_accuracy = np.mean(R_errors < 0.5)
    R_5_accuracy = np.mean(R_errors < 5)
    T_5_accuracy = np.mean(T_errors < 5)
    R_10_accuracy = np.mean(R_errors < 20)
    T_10_accuracy = np.mean(T_errors < 20)
    R_median = np.median(R_errors)
    T_median = np.median(T_errors)
    err_median = np.median(np.array(point_errs))
    err_ransac_median = np.median(np.array(point_errs_masked))
    aucs = pose_auc(R_errors, [5, 10, 20])
    aucs = [100.*yy for yy in aucs]
    print('R_0.5_accuracy: {}, max_5_accuracy: {}, min_5_accuracy: {}, max_20_accuracy: {}, min_20_accuracy: {},'
          ' max_median: {}, min_median: {}, err_median: {}, err_ransac_median: {}'
          .format(np.round(R_2_accuracy, 4), np.round(R_5_accuracy, 4), np.round(T_5_accuracy, 4),
                  np.round(R_10_accuracy, 4), np.round(T_10_accuracy, 4),
                  np.round(R_median, 4), np.round(T_median, 4),
                  np.round(err_median, 4), np.round(err_ransac_median, 4)))
    print('{:.3}/{:.3}/{:.3}'.format(aucs[0], aucs[1], aucs[2]))
    return results

def Get_resize_ratio(shape_origin, shape):
    w = shape_origin[0].astype(float)
    h = shape_origin[1].astype(float)
    w_new = shape[0].astype(float)
    h_new = shape[1].astype(float)
    h_w = h_new / w_new
    ratio = -1
    add_num = [0, 0]
    if w / w_new < h / h_new:
        ratio = w_new / w
        add_num[1] = (h - w * h_w) / 2
    else:
        ratio = h_new / h
        add_num[0] = (w - h / h_w) / 2
    return ratio, np.array(add_num)



def scale_intrinsics(K, scales):
    scales = np.diag([1./scales[0], 1./scales[1], 1.])
    return np.dot(scales, K)

def Get_params(file_name):
    f = open(file_name)
    line = f.readline()
    name_list = {}
    size_list = []
    K_list = []
    P_list = []
    i = 0
    while line:
        data = np.array(line.split())
        name_list[data[0]] = i
        size_list.append([data[1], data[2]])
        K = np.identity(3)
        K[0, 0] = data[3]
        K[1, 1] = data[4]
        K[0, 2] = data[5]
        K[1, 2] = data[6]
        K_list.append(K)
        P = np.zeros((3, 4))
        P[0:3, 0:4] = data[7:19].astype(np.float64).reshape(3, 4)
        P_list.append(P)
        i += 1
        line = f.readline()
    return (name_list, size_list, K_list, P_list)


def Get_pairs(file_name):
    f = open(file_name)
    line = f.readline()
    pair_list = []
    while line:
        data = np.array(line.split())
        pair_list.append(data)
        line = f.readline()
    return np.array(pair_list)


def Get_all_pairs(path, use_set='train'):
    all_pairs = np.array([])
    if use_set=="train":
        set_file = path + "megadepth_train_scenes.txt"
    if use_set=="test":
        set_file = path + "megadepth_validation_scenes_new.txt"
    if use_set=="test2":
        set_file = path + "megadepth_validation_scenes_old.txt"
    if use_set=="test_full":
        set_file = path + "megadepth_validation_scenes_full.txt"
    f = open(set_file, 'r')
    all_pairs = np.zeros([0, 4])
    F_list = np.zeros([0, 3, 3])
    lines = np.array(f.readlines())
    length = lines.shape[0]
    # if use_set=="train":
    #     length = 1
    for i in range(length):
    # for i in range(1):
        file_name_2 = np.sort(np.array(os.listdir(path + lines[i].strip('\n'))))
        for j in range(file_name_2.shape[0]):
            the_pair = Get_pairs(path + lines[i].strip('\n') + "/" + file_name_2[j] + "/pair_list.txt").reshape(-1, 2)
            the_F = np.load(path + lines[i].strip('\n') + "/" + file_name_2[j] + "/epipolar.npy").reshape(-1, 3, 3)
            type_name = np.array([lines[i].strip('\n') + "/" + file_name_2[j]]).reshape(1, 1).repeat(the_pair.shape[0], axis=0)
            sequence = np.arange(the_pair.shape[0]).reshape(-1, 1).astype(str)
            # print(type_name.shape, sequence.shape, the_pair.shape)
            the_pair = np.concatenate((type_name, sequence, the_pair), axis=1)
            all_pairs = np.concatenate([all_pairs, the_pair], axis=0)
            F_list = np.concatenate([F_list, the_F], axis=0)
    return np.array(all_pairs), np.array(F_list)



def Resize_depth(depth, shape=np.array([640, 480])):
    w = depth.shape[1]
    h = depth.shape[0]
    w_new = shape[0]
    h_new = shape[1]
    if w/w_new < h/h_new:
        gap = int((h - w / w_new * h_new) / 2)
        crop_depth = depth[gap:h - gap, :]
    else:
        gap = int((w - h / h_new * w_new) / 2)
        crop_depth = depth[:, gap:w - gap]
    resize_depth = cv2.resize(
        crop_depth, tuple(shape), interpolation=cv2.INTER_LINEAR)
    return resize_depth


def Resize_img(depth, shape=np.array([640, 480])):
    w = depth.shape[1]
    h = depth.shape[0]
    w_new = shape[0]
    h_new = shape[1]
    if w/w_new < h/h_new:
        gap = int((h - w / w_new * h_new) / 2)
        crop_img = depth[gap:h - gap, :, :]
    else:
        gap = int((w - h / h_new * w_new) / 2)
        crop_img = depth[:, gap:w - gap, :]
    resize_img = cv2.resize(
        crop_img, tuple(shape))
    return resize_img





def loss_function_matches(scores, gt_matches, if_success, if_nomatching1_gt, confidence):
    position = gt_matches.reshape(-1, gt_matches.shape[1], 1).long()
    x = torch.gather(1 - scores, 2, torch.where(position.long() == scores.shape[1], torch.tensor(0, device=scores.device), position.long()))
    x = x.reshape(-1, gt_matches.shape[1])
    weight = torch.clamp((if_success.float().sum(1) + 30) / (torch.logical_not(if_nomatching1_gt).float().sum(1) + 30), 0.3, 1.0)
    x = torch.where(if_success, torch.tensor(0.0, device=scores.device), x) / (if_success.float().sum(1)[:, None] + 1) / scores.shape[0] / weight[:, None]
    loss_matches = x.sum()
    loss_nomatches = (torch.where(if_nomatching1_gt, -(1 - confidence + 1e-9).log(), 
        -(confidence + 1e-9).log()) / (if_nomatching1_gt.shape[0] * if_nomatching1_gt.shape[1])).sum()
    return loss_matches, loss_nomatches
    

def Position_loss(label, project_results, project_results_all, noc, batch_normalize=True):
    zero = torch.tensor(0.0, device=label.device)
    distance_loss = ((project_results[..., :2] - label.reshape(project_results.shape)[..., :2]).pow(2).sum(3) + 1e-10).sqrt()
    # distance_loss2 = ((project_results_all[:, :, :, :, :2] - label.reshape(project_results.shape)[:, :, :, None, :2]).pow(2).sum(4) + 1e-10).sqrt()
    criterion = torch.logical_and(noc.reshape(project_results.shape[0], project_results.shape[1], project_results.shape[2]),
        label.reshape(project_results.shape)[:, :, :, 0] > -0.01)
    # num_new_satisfy = criterion.float().sum()
    # position_loss = loss_matches
    average_error = (project_results[..., :2] - label.reshape(project_results.shape)[..., :2]).pow(2).max(3)[0].sqrt().detach()
    median = torch.zeros(label.shape[0], device=label.device) + 10
    for j in range(label.shape[0]):
        if criterion[j].float().sum() > 0.5:
            median[j] = average_error[j][criterion[j]].median()
    if criterion.float().sum() > 0.5:
        median_all = torch.median(average_error[criterion])
    else:
        median_all = torch.tensor(10.0, device=label.device)
    # if i == 4:
    #     print(i, median.mean())
    # print(average_error[criterion].max())
    if batch_normalize:
        threshold =  median[:, None, None]
    else:
        threshold = median_all
    criterion = torch.logical_and(average_error < 3 * threshold, criterion)
    # criterion2 = torch.logical_and(criterion[:, :, :, None].expand(-1, -1, -1, 9), distance_loss2 < 1.5 * distance_loss[:, :, :, None].expand(-1, -1, -1, 9))
    # criterion2 = torch.logical_and(criterion2, distance_loss2 < 8)
    distance_loss = torch.where(criterion, distance_loss,
        torch.tensor(0.0, device=label.device)) / median_all.pow(2).detach() * 100
    # distance_loss2 = torch.where(criterion2, distance_loss2, torch.tensor(0.0, device=label.device)) / median_all.detach() * 10
    # distance_loss = torch.where(average_error < 16.0, distance_loss * 2, distance_loss)
    # distance_loss = torch.where(average_error < 8.0, distance_loss * 2, distance_loss)
    # distance_loss = torch.where(average_error < 4.0, distance_loss * 2, distance_loss)
    # distance_loss = torch.where(average_error < 2.0, distance_loss * 2, distance_loss)
    # distance_loss = torch.where(average_error < 1.0, distance_loss * 2, distance_loss)
    position_loss = distance_loss / (criterion.float().sum(2).sum(1)[:, None, None] + 5) / 32.0
    # position_loss2 = torch.where(criterion2, distance_loss2, torch.tensor(0.0, device=label.device)) / (criterion2.float().reshape(label.shape[0], -1).sum(1)[:, None, None, None] + 5) / 32
    position_loss = position_loss.sum()
    # position_loss = torch.where(criterion, distance_loss, zero) / criterion.float().sum()
    # print(median, position_loss.sum())
    return position_loss, average_error, average_error


def Segment_loss(project_results, seg, label, if_matching1):
    seg_use = seg.permute(0, 2, 3, 1)
    if_suppress, if_choose, distance = Compare_projects(project_results, label)
    ce_loss = F.binary_cross_entropy_with_logits(seg_use, if_choose, reduction="none")
    p_t = seg_use * if_choose + (1 - seg_use) * (1 - if_choose)
    loss =(1 - p_t).pow(2) * ce_loss
    alpha_t = if_choose + (1 - if_choose) / ((if_suppress > 0.5).float().sum()) * (if_choose > 0.5).float().sum()
    seg_loss = alpha_t * loss
    seg_loss = seg_loss[torch.logical_or(if_suppress > 0.5, if_choose > 0.5)].sum() / (label[:, :, :, 0] > -0.01).float().sum() * 100    
    seg_scores, seg_result = seg_use.max(3)
    distance_choose = torch.gather(distance, 3, seg_result[:, :, :, None]).squeeze(3)
    b, _, h, w = if_matching1.shape
    if_matching1_use = if_matching1.permute(0, 2, 3, 1)[:, :, None, :, None].expand(-1, -1, 4, -1, 4, -1).reshape(b, h * 4, w * 4, 9)
    if_matching1_use = torch.gather(if_matching1_use, 3, seg_result[:, :, :, None]).squeeze(3)
    criterion = torch.logical_and(if_matching1_use > 0.5, label[:, :, :, 0] > -0.01)
    choice_rate = (distance_choose[criterion] < 5.0).float().sum() / (criterion.float().sum() + 1e-9)
    return seg_loss, choice_rate

def Compute_positions_and_ranges(height, width, device):
    cols = torch.arange(0, height).reshape(height, 1).repeat(1, width).reshape(width * height)
    rows = torch.arange(0, width).reshape(1, width).repeat(height, 1).reshape(width * height)
    positions = torch.zeros((width * height, 2), device=device)
    positions[:, 0] = cols
    positions[:, 1] = rows
    max_shape = max(height, width)
    ranges = torch.zeros([max_shape, max_shape], device=device)
    for i in range(max_shape):
        ranges[i] = F.pad(torch.arange(i + 1), [0, max_shape - 1 - i], "constant", 1e7)
    return positions, ranges

def choice_loss(scores, error):
    min_error, indices = error.min(1)
    # scores = F.softmax(scores, dim=1)
    indices_one_hot = torch.zeros_like(scores).float()
    indices_one_hot = torch.scatter(indices_one_hot, 1, indices[:, None].long(), torch.ones_like(indices[:, None].float()))
    loss = torch.where(indices_one_hot > 0.5, -(scores + 1e-9).log() * 8,
        - (error - min_error[:, None]) * (1 - scores + 1e-9).log()) / scores.shape[0]
    return loss.sum()

def loss_step(label1, scores, indices_medium, confidence, height, width, patch_scale, H_matrix, gather_homography, i, left=None, right=None):
    project_results, grid = project_function(H_matrix, width, height, 32)
    # segment_loss1 = Segment_loss(project_results, seg, label1)
    gt_use = label1.reshape(project_results.shape[0], project_results.shape[1], project_results.shape[2], -1)[:, :, :, None,
        :].expand(-1, -1, -1, project_results.shape[3], -1)
    distance = (project_results[:, :, :, :, :2] - gt_use[:, :, :, :, :2]).pow(2).sum(4).sqrt()
    distance[:, :, :, [0, 1, 2, 3, 5, 6, 7, 8]] *= 2
    indices_zero_shot_local = torch.ones([distance.shape[0], height, width, patch_scale // 2 * patch_scale // 2], device=scores.device)
    min_distance_without_local = distance[:, :, :, 4]
    min_distance_without_local = torch.where(label1[:, :, 0].reshape(scores.shape[0], height * patch_scale // 2, width * patch_scale // 2) < -1,
        torch.tensor(10000.0, device=scores.device), min_distance_without_local)
    _, indices_without_local =  (min_distance_without_local.reshape(distance.shape[0], height, 
        patch_scale // 2, width, patch_scale // 2).permute(0, 1, 3, 2, 4)\
        .reshape(distance.shape[0], height, width, -1) + (grid.flatten(3, 4)[..., :2] % 32 - 16).pow(2).sum(4).sqrt()).min(3)
    indices_zero_shot_local = torch.scatter(indices_zero_shot_local, 3, indices_without_local[:, :, :, None], 
        torch.zeros([distance.shape[0], height, width, 1], device=scores.device)).reshape(distance.shape[0], 
        height, width, patch_scale // 2, patch_scale // 2).permute(0, 1, 3, 2, 4).reshape(distance.shape[0], 
        distance.shape[1], distance.shape[2])
    min_distance, indices = torch.min(distance, 3)
    # min_distance = distance[..., 4]
    # indices[...] = 4

    label1_new = label1.reshape(scores.shape[0], height, patch_scale//2, width, patch_scale//2, 3).permute(0, 1, 3, 2, 4, 
        5).reshape(scores.shape[0], height, width, patch_scale * patch_scale // 4, 3)
    label1_new = torch.gather(label1_new, 3, indices_without_local[:, :, :, None, None].repeat(1, 1, 1, 1, 3))[:, :, :, 0]
    # tgt_pos = label1_new.clone()

    if_nomatching1_gt = (label1_new[:, :, :, 0] < -0.1).reshape(scores.shape[0], -1)
    label1_new = (label1_new[:, :, :, 0].int() // patch_scale + label1_new[:, :, :, 1].int() // patch_scale * width).reshape(scores.shape[0], -1)
    noc_base = torch.logical_and((label1_new % width - indices_medium % width).abs() < 1.5, 
        (label1_new // width - indices_medium // width).abs() < 1.5).reshape(scores.shape[0], 1, height, width).expand(-1, 9, -1, -1).float()
    noc = gather_homography(noc_base).permute(0, 2, 3, 1).reshape(scores.shape[0], height, 1, width, 1, 9, 9)[:, :, :, :, :, :, 0].expand(-1, -1, 
        patch_scale // 2, -1, patch_scale // 2, -1).reshape(label1.shape[0], height * patch_scale // 2, width * patch_scale // 2, 9)
    label1_new = torch.where(if_nomatching1_gt, torch.tensor(width * height, device=scores.device).int(), label1_new)
    # if i == 4:
    #     print((torch.logical_and(noc_base[:, 0, :, :].reshape(scores.shape[0], -1) > 0.5, label1_new != (width *\
    #         height)).float().sum(1) / ((label1_new != (width * height)).float().sum(1) + 1e-9)).mean(),
    #         ((confidence > 0.5) == torch.logical_not(if_nomatching1_gt)).float().sum(1).mean() / (confidence.shape[1]), 
    #         (indices[noc[:, :, :, 4] > 0.5] == 4).sum() / ((noc[..., 4] > 0.5).float().sum() + 1e-7))
    grid_x, grid_y = torch.meshgrid(torch.arange(height, device=scores.device), torch.arange(width, device=scores.device))
    grid_new = torch.stack([grid_y, grid_x], -1).reshape(1, 1, -1, 2).expand(scores.shape[0], height * width, -1, -1)
    # if left != None and i ==4:
    #     src_choice = indices_without_local[tgt_pos[:, :, :, 0] > -0.1]
    #     src_average_dis = (((src_choice // 16).float() - 7.5).pow(2) + ((src_choice % 16).float() - 7.5).pow(2)).sqrt()
    #     src_choice2 = indices_without_local[torch.logical_and(tgt_pos[:, :, :, 0] > -0.1, noc_base[:, 0] < 0.5)]
    #     src_average_dis2 = (((src_choice2 // 16).float() - 7.5).pow(2) + ((src_choice2 % 16).float() - 7.5).pow(2)).sqrt()
    #     print("average_distance_to_center", src_average_dis.mean(), src_average_dis2.mean())
    #     left_show = left[0].cpu().numpy()
    #     right_show = right[0].cpu().numpy()
    #     src_points = torch.cat([indices_without_local[0, :, :, None] % 16 + 16 * grid_new[0, 0, :, 0].reshape(height, width, 1), 
    #         indices_without_local[0, :, :, None] // 16 + 16 * grid_new[0, 0, :, 1].reshape(height, width, 1)], dim=2)
    #     src_points = src_points[tgt_pos[0, :, :, 0] > -0.1].int().cpu().numpy() * 2
    #     tgt_pos_use = tgt_pos[0][tgt_pos[0, :, :, 0] > -0.1].cpu().numpy()[:, :2]
    #     for j in range(src_points.shape[0]):
    #         colar = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
    #         cv2.circle(left_show, (src_points[j][0], src_points[j][1]), 2, colar)
    #         cv2.circle(right_show, (int(tgt_pos_use[j][0]), int(tgt_pos_use[j][1])), 2, colar)
    #     for x in range(0, width * 32, 32):
    #         cv2.line(left_show, (x, 0), (x, height * 32), color=(0, 255, 0), thickness=1)        
    #     for y in range(0, height * 32, 32):
    #         cv2.line(left_show, (0, y), (width * 32, y), color=(0, 255, 0), thickness=1)
    #     result0 = cv2.hconcat([left_show, right_show])
    #     cv2.imwrite("/home/junjieni/test/all/" + str(np.random.randint(100000)) + ".jpg", result0.astype(np.uint8)[:, :, [2, 1, 0]])
    #     left_show = left[0].cpu().numpy()
    #     right_show = right[0].cpu().numpy()
    #     src_points = torch.cat([indices_without_local[0, :, :, None] % 16 + 16 * grid_new[0, 0, :, 0].reshape(height, width, 1), 
    #     indices_without_local[0, :, :, None] // 16 + 16 * grid_new[0, 0, :, 1].reshape(height, width, 1)], dim=2)
    #     src_points = src_points[torch.logical_and(tgt_pos[0, :, :, 0] > -0.1, noc_base[0, 0] < 0.5)].int().cpu().numpy() * 2
    #     tgt_pos = tgt_pos[0][torch.logical_and(tgt_pos[0, :, :, 0] > -0.1, noc_base[0, 0] < 0.5)].cpu().numpy()[:, :2]
    #     for j in range(src_points.shape[0]):
    #         colar = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
    #         cv2.circle(left_show, (src_points[j][0], src_points[j][1]), 2, colar)
    #         cv2.circle(right_show, (int(tgt_pos[j][0]), int(tgt_pos[j][1])), 2, colar)
    #     for x in range(0, width * 32, 32):
    #         cv2.line(left_show, (x, 0), (x, height * 32), color=(0, 255, 0), thickness=1)        
    #     for y in range(0, height * 32, 32):
    #         cv2.line(left_show, (0, y), (width * 32, y), color=(0, 255, 0), thickness=1)
    #     result1 = cv2.hconcat([left_show, right_show])
    #     cv2.imwrite("/home/junjieni/test/fail/" + str(np.random.randint(100000)) + ".jpg", result1.astype(np.uint8)[:, :, [2, 1, 0]])
    noc_base_final = torch.logical_and((label1_new[:, :, None] % width - grid_new[:, :, :, 1]).abs() < 1.5, 
        (label1_new[:, :, None] // width - grid_new[:, :, :, 0]).abs() < 1.5).reshape(scores.shape[0], height * width, -1)
    scores_gap = scores.clone().detach()
    scores_gap = scores - torch.where(noc_base_final, torch.zeros_like(scores) + 0.05, 
        torch.zeros_like(scores))
    indices_final_use = torch.argmax(scores_gap, dim=2)
    noc_base_final = torch.logical_and(torch.logical_and((label1_new % width - indices_final_use % width).abs() < 1.5, 
        (label1_new // width - indices_final_use // width).abs() < 1.5), label1_new!=height * width).reshape(scores.shape[0], 
        1, height, width).expand(-1, 9, -1, -1).float()
    indices = torch.where(min_distance <= 32, indices, torch.tensor(4, device=project_results.device))
    noc = torch.gather(noc, 3, indices[:, :, :, None]) > 0.5
    # choice = seg[:, :-1, :, :].max(1)[1]
    # project_results_choice = torch.gather(project_results, 3, choice[:, :, :, None, None].repeat(1, 1, 1, 1, 3))[:, :, :, 0]
    project_results_optimal = torch.gather(project_results, 3, indices[:, :, :, None, None].repeat(1, 1, 1, 1, 3))[:, :, :, 0]
    # project_results_optimal = project_results[:, :, :, 4]
    # true_choice = (choice == indices)
    # mismatching_loss1 = Continuous_loss(if_nomatching1, width, height).multiply(scores[:, :-1, -1])
    position_loss1, distance1, average_error1 = Position_loss(label1, project_results_optimal, project_results, noc)
    # average_error2 = (project_results_choice[:, :, :, :2] - label1.reshape(project_results[:, :, :, 4].shape)[:, :, :, :2]).pow(2).sum(3).sqrt()
    position_loss1 = position_loss1.sum()
    # label_new_oneshot = torch.zeros(scores.shape, device=x.device).float()
    # label_new_oneshot = torch.scatter(label_new_oneshot, 2, label1_new[:, :, None].long(), torch.ones_like(label1_new[:, :, None]).float())
    classifying_loss, nomatching_loss = loss_function_matches(scores, label1_new, 
        torch.logical_or((noc_base_final[:, 0, :] > 0.5).reshape(scores.shape[0], -1), if_nomatching1_gt), if_nomatching1_gt, confidence)
    return position_loss1, classifying_loss, nomatching_loss, average_error1, \
        project_results.reshape(scores.shape[0], height, patch_scale//2, 
        width, patch_scale//2, 9, 3).permute(0, 1, 3, 2, 4, 5, 6)[:, :, :, [3, 3, 11, 11], [3, 11, 3, 11], 4, :2].reshape(scores.shape[0], 
        height * width, 4, 2), \
        grid[:, :, :, [3, 3, 11, 11], [3, 11, 3, 11], :2].reshape(scores.shape[0], height * width, 4, 2), if_nomatching1_gt, project_results



def Compute_loss(results, label1, label2, left, right, reverse_set=False, width=20, height=15, patch_scale=32,
                 iter=15, refine_mode=False, eval_scores=None, loss_type='distance', if_refine=False, if_choose=True, gather_homography=None,
                 left_feature=None, right_feature=None):
    if refine_mode==False:
        if_matching1 = (results[-1]["confidence"] > 0.4)
        width = right.shape[2] // patch_scale
        height = right.shape[1] // patch_scale
        position_loss = torch.tensor(0.0, device=label1.device)
        classifying_loss = torch.tensor(0.0, device=label1.device)
        position_loss_refine = torch.tensor(0.0, device=label1.device)
        nomatching_loss = torch.tensor(0.0, device=label1.device)
        if_matching1_new = gather_homography(if_matching1.float().reshape(-1, 1, height, width).expand(-1, 9, -1, -1)).reshape(-1, 9, 9, height, width)[:, :, 0]
        for i in range(1):
            result = results[i]
            position_loss_step, classifying_loss_step, nomatching_loss_step, average_error1, project_results, grid, if_nomatching1_gt, project_results_all =\
                loss_step(label1, result["scores"], result["indices_medium"], result["confidence"], height, width, patch_scale, result["H_matrix"],
                gather_homography, i, left, right)
            position_loss += position_loss_step
            classifying_loss += classifying_loss_step
            nomatching_loss += nomatching_loss_step
        result = results[-1]
        matches_l = grid[if_matching1].reshape(-1, 2)
        matches_r = project_results[if_matching1].reshape(-1, 2)
        # accurate = (if_nomatching1_gt == torch.logical_not(if_matching1))
        label_use = label1
        projects_use = project_results_all.reshape(-1, height, 4, 4, width, 4, 4, 9, 3)[:, :, :, 2, :, :, 2].reshape(-1, height * 4, width * 4, 9, 3)
        label_use2 = label1.reshape(-1, height, 4, 4, width, 4, 4, 3)[:, :, :, 2, :, :, 2].reshape(-1, height * 4, width * 4, 3)
        confidence2 = results[-1]["confidence2"]
        label_refine = (label_use2[result['if_matching']][:, :2] - result['mkpts1_c'][:, :2] + 6)
        if_matching_refine_gt = torch.logical_not(torch.logical_or((label_refine < 0).float().sum(1) > 0.5, (label_refine > 12).float().sum(1) > 0.5))
        label_refine_grid = torch.round(label_refine / 2.0).long()
        label_refine_grid = label_refine_grid[:, 0] + label_refine_grid[:, 1] * 7
        # if_classify_loss_refine = torch.logical_and(if_matching_refine_gt, torch.logical_not(if_success))
        seg_loss, choice_rate = Segment_loss(projects_use, result["scores_segment"], label_use2, if_matching1_new)
        position_loss_refine_base = (label_use2[result['if_matching']][:, :2] - result['mkpts1_f'][:, :2]).pow(2).sum(1)
        if_success = (position_loss_refine_base < 2.5)
        if_position_loss_refine = torch.logical_and(if_matching_refine_gt, if_success)
        position_loss_refine = torch.where(if_matching_refine_gt, position_loss_refine_base.sqrt(), 
            torch.zeros_like(position_loss_refine_base)) / (if_matching_refine_gt.float().sum() + 5)
        # classifying_loss_refine = torch.gather(result['refine_scores'][if_classify_loss_refine], 1, 
        #     label_refine_grid[if_classify_loss_refine][:, None])
        # print((if_position_loss_refine.float().sum() / (if_matching_refine_gt.float().sum() + 1e-7)).cpu(), 
        #     classifying_loss_refine.median(), position_loss_refine_base.sqrt()[if_position_loss_refine].mean())
        # classifying_loss_refine = - torch.clamp(classifying_loss_refine, 1e-7, 1-1e-7).log().sum() / (if_matching_refine_gt.float().sum() + 1)
        confidence_loss_refine = torch.where(if_position_loss_refine, -(confidence2 + 1e-9).log(), -(1- confidence2 + 1e-9).log()).mean()
        accurate = (if_position_loss_refine == results[-1]["if_matching2"])
        Medium_information = {
            # 'choice_rate': choice_rate.cpu(),
            'choice_rate': (if_position_loss_refine.float().sum() / (if_matching_refine_gt.float().sum() + 1e-7)).cpu(),
            'classify_accurate': accurate.cpu(),
            'average_error1': average_error1.cpu(),
            # 'matches_r': matches_r.cpu(),
            # "matches_l": matches_l.cpu(),
            'matches_r': results[-1]['mkpts1_f'][results[-1]['if_matching2']].cpu(),
            "matches_l": results[-1]['mkpts0_f'][results[-1]['if_matching2']].cpu(),
            "label_use": label_use.cpu(),
        }
    else: 
        if_matching1 = results[-1]["if_matching"]
        # if_matching2 = results[-1]["confidence"] > 0.5
        project_results = results[-1]["project_test"]
        indices = results[-1]["indices"]
        tgt_positions = project_results.reshape(if_matching1.shape[0], height * 4, 4, width * 4, 4, 9, 3)[:, :, 2, :, 2, :, :2].\
            reshape(if_matching1.shape[0], height, 4, width, 4, 9, 2).permute(0, 1, 3, 2, 4, 5, 6).flatten(1, 2).flatten(2, 3)
        width = right.shape[2] // patch_scale
        height = right.shape[1] // patch_scale
        position_loss = torch.tensor(0.0, device=label1.device)
        classifying_loss = torch.tensor(0.0, device=label1.device)
        nomatching_loss = torch.tensor(0.0, device=label1.device)        
        label_use = label1.reshape(label1.shape[0], height, 4, 4, width, 4, 4, 3)[:, :, :, 2, :, :, 2].permute(0, 1, 3, 2, 4, 5).flatten(1, 2).flatten(2, 3)
        # test_results = (tgt_positions - label_use[:, None, :2]).pow(2).sum(2)[:, 4][label_use[:, 0] > -0.0001]
        classifying_loss = choice_loss(results[-1]["choice_scores"][label_use[:, :, :, 0] > - 0.001], 
            (tgt_positions - label_use[:, :, :, None, :2]).pow(2).sum(4).sqrt()[label_use[:, :, :, 0] > - 0.001].detach()) * 1000
        label_use = label_use[if_matching1]
        tgt_positions = torch.gather(tgt_positions, 3, indices[:, :, :, None, None].expand(-1, -1, -1, -1, 2))[:, :, :, 0][if_matching1]
        # for i in range(1):
        #     result = results[i]
        #     confidence = result["confidence"]
        #     loss_nomatches = (torch.where(torch.logical_not(if_matching2), -(1 - confidence + 1e-9).log(), 
        #         -(confidence + 1e-9).log()) / (if_matching2.shape[0] + 1e-9)).sum()
        #     position_loss_step, _, average_error1 =\
        #         Position_loss(label_use[:, None, None, :2], result["tgt_positions"][:, None, None], if_matching2[:, None, None], False)
        #     position_loss += position_loss_step.sum()
        #     nomatching_loss += loss_nomatches
        # matches_l = results[-1]["ref_positions"]
        # matches_r = results[-1]["tgt_positions"]
        # accurate = ((label_use > -1e-9) == if_matching2)
        # if_matching1 = torch.logical_and(if_matching1, torch.logical_not(if_nomatching1_gt.reshape(label1.shape[0], -1)))
        # Medium_information = {
        #     'classify_accurate': accurate,
        #     'average_error1': average_error1,
        #     'matches_r': matches_r,
        #     "matches_l": matches_l,
        #     "label_use": label_use,
        # }
        Medium_information = {
            'classify_accurate': torch.zeros_like(label_use).bool(),
            'average_error1': torch.zeros_like(label_use[:, None, None, 0]),
            'matches_r': torch.zeros_like(results[-1]["ref_positions"]),
            "matches_l": torch.zeros_like(results[-1]["ref_positions"]),
            "label_use": label_use,
        }
    
    Loss = {
        'position': position_loss,
        'classify': classifying_loss,
        'nomatching': nomatching_loss,
        'seg_loss': seg_loss,
        'position_refine': position_loss_refine.sum(),
        'confidence_refine': confidence_loss_refine
    }
    return (Loss, Medium_information)

def project_function(H_matrix, width, height, patch_size_high, step=2):
    grid_x, grid_y = torch.meshgrid(torch.arange(0, height * patch_size_high, step, device=H_matrix.device) + 1.0, torch.arange(0, 
        width * patch_size_high, step, device=H_matrix.device) + 1.0)
    grid = torch.stack([grid_y, grid_x, torch.ones_like(grid_y)], -1).reshape(height, patch_size_high//step, width, patch_size_high//step, 3).permute(0, 2, 1, 3, 4)[None].\
        expand(H_matrix.shape[0], -1, -1, -1, -1, -1)
    # (b, h, w, proposals, 3, 3),(b, h, w, size, size, 3)->(b, h, w, size, size, proposals, 3)
    project_results = torch.einsum('abckij,abcdej->abcdeki', H_matrix, grid).permute(0, 1, 3, 2, 4, 5, 6).\
        reshape(grid.shape[0], height*patch_size_high // step, width*patch_size_high // step, H_matrix.shape[3], 3)
    project_results[:, :, :, :, 2] = torch.where(project_results[:, :, :, :, 2].abs() > 1e-3, project_results[:, :, :, :, 2], 
        torch.ones_like(project_results[:, :, :, :, 2]))
    project_results = project_results / project_results[:, :, :, :, 2, None].expand(-1, -1, -1, -1, 3)
    # project_results = torch.where(torch.logical_or(torch.isnan(project_results), torch.isinf(project_results)), 
    #     torch.zeros_like(project_results), project_results)
    # print(project_results.sum())
    # print(H_matrix[0, 0, 0, 4], project_results[0, 10, 10, 4])
    return project_results, grid

def Compare_projects(project_results, gt_project_results, threshold_low=3.0, threshold_high=5.0):
    # project_results: (b, h//8, w//8, 9, 3)
    # gt_project_results: (b, h//8, w//8, 3)
    if_suppress = torch.zeros_like(project_results[:, :, :, :, 0])
    if_choose = torch.zeros_like(project_results[:, :, :, :, 0])
    gt_use = gt_project_results.reshape(project_results.shape[0], project_results.shape[1], project_results.shape[2], -1)[:, :, :, None,
        :].expand(-1, -1, -1, project_results.shape[3], -1)
    distance = (project_results[:, :, :, :, :2] - gt_use[:, :, :, :, :2]).abs().max(4)[0]
    min_distance, indices = torch.min(distance, 3)
    # indices[gt_use[:, :, :, 0, 0] < -0.01] = 9
    if_choose.scatter_(3, indices[:, :, :, None].long(), torch.ones(indices.shape, device=distance.device)[:, :, :, None].float())
    if_suppress = torch.where(distance < threshold_high, if_suppress, if_suppress + 1)
    if_suppress = torch.where(gt_use[:, :, :, :, 0] < -0.01, torch.zeros_like(project_results[:, :, :, :, 0]), if_suppress)
    if_choose = torch.where(torch.logical_or(gt_use[:, :, :, :, 0] < -0.01, distance > threshold_low),
        torch.zeros_like(project_results[:, :, :, :, 0]), if_choose)
    return if_suppress, if_choose, distance