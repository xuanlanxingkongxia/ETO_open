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
from scipy.spatial.transform import Rotation as R
sys.path.append(r"/home/junjieni/research/FDesp/")
from utils.utils import Resize_depth, Get_pairs, Get_params, Create_P_resize, Resize_img, Get_resize_ratio
import multiprocessing as mp
import torchvision.models as models
from models.first_layer import feature_extractor
from utils.coco_utils import compute_F

def Read_colmap_files(path):
    # camera param: CAMERA_ID, WIDTH, HEIGHT, FOCAL_LENGTH, CX, CY
    # image param: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID
    # point param: POINT3D_ID, X, Y, Z, R, G, B, ERROR
    camera_file = open(path + "cameras.txt", 'r')
    image_file = open(path + "images.txt", 'r')
    point_file = open(path + "points3D.txt", 'r')
    cameras = []
    images = []
    images_id = {}
    cameras_id = {}
    points_id = {}
    points = []
    for j in range(4):
        line = camera_file.readline()
    i = 0
    while line:
        line_list = np.array(line.split())
        index = np.array([0, 2, 3, 4, 5, 6, 7])
        camera = line_list[index].astype(float)
        cameras_id[str(line_list[0])] = i
        i += 1
        cameras.append(camera)
        line = camera_file.readline()
    for j in range(5):
        line = image_file.readline()
    i = 0
    while line:
        line_list = np.array(line.split())
        image = {"params": line_list[0:9].astype(float), "name": line_list[9]}
        images_id[str(line_list[0])] = i
        i += 1
        line = image_file.readline()
        line_list = np.array(line.split()).astype(float)
        image["keypoints"] = line_list.reshape(-1, 3)
        images.append(image)
        line = image_file.readline()
    for j in range(4):
        line = point_file.readline()
    i = 0
    while line:
        line_list = np.array(line.split())
        point = {"params": line_list[0:8].astype(float), "track": line_list[8:].reshape(-1, 2).astype(int)}
        points.append(point)
        points_id[str(line_list[0])] = i
        i += 1
        line = point_file.readline()
    image_file.close()
    point_file.close()
    camera_file.close()
    return (np.array(cameras), np.array(images), np.array(points), images_id, cameras_id, points_id)




def Save_image_parameters(pairs, images, cameras, save_path, cameras_id):
    f_p = open(save_path + "pair_list.txt", 'w')
    f_i = open(save_path + "img_cam.txt", 'w')
    for (i, pair) in enumerate(pairs):
        name1 = images[pair[0]]["name"]
        name2 = images[pair[1]]["name"]
        f_p.write(name1 + " " + name2 + "\n")
    f_p.close()
    img_set = set(np.array(pairs).reshape(-1).tolist())
    for img in img_set:
        name = images[img]["name"]
        camera_id = images[img]["params"][8]
        camera_position = cameras_id[str(int(camera_id))]
        width = cameras[camera_position][1]
        height = cameras[camera_position][2]
        f = cameras[camera_position][3]
        cx = cameras[camera_position][4]
        cy = cameras[camera_position][5]
        quaternion = np.zeros(4)
        quaternion[0:3] = images[img]["params"][2:5]
        quaternion[3] = images[img]["params"][1]
        transform = np.array(images[img]["params"][5:8]).reshape(3, 1)
        r = np.array(R.from_quat(quaternion).as_matrix())
        p_matrix = np.concatenate((r, transform), axis=1).reshape(-1)
        output = name + " " + str(int(width)) + " " + str(int(height)) + " " + str(f) + " " + str(f) + " " + \
                 str(cx) + " " + str(cy) + " " + " ".join(str(i) for i in p_matrix) + "\n"
        f_i.write(output)
    f_i.close()


def Megadepth_preprocess(path, depth_path, save_path):
    depth_names = os.listdir(depth_path)
    depth_name = [os.path.splitext(x)[0] + ".jpg" for x in depth_names]
    depth_name = set(depth_name)
    # pair_list = Get_pairs(save_path + "pair_list.txt")
    # i = 1
    # p = os.path.splitext(pair_list[1, 0])[0]
    # q = os.path.splitext(pair_list[1, 1])[0]
    # depth0 = h5py.File(depth_path + p + ".h5", "r")['depth']
    # depth1 = h5py.File(depth_path + q + ".h5", "r")['depth']
    # test_pair_points(0, 0, pair_list, 0, i, save_path, depth0, depth1)
    cameras, images, points, images_id, cameras_id, points_id = Read_colmap_files(path)
    pairs = Compute_pairs_foundation(points, images_id, images.shape[0])
    print(pairs.shape)
    pairs_new = Check_pair_depth_exist(pairs, images, depth_name)
    print(pairs_new.shape)
    # test_pair_points(points, points_id, pairs_new, images, i, save_path, depth0, depth1)
    Save_image_parameters(pairs_new, images, cameras, save_path, cameras_id)

def Create_files(path):
    if_exists = os.path.exists(path)
    if not (if_exists):
        os.makedirs(path)


def Auto_process_script(path, depth_path, save_path):
    folder_name_1 = np.sort(np.array(os.listdir(depth_path)))
    depth_path_1 = np.array([os.path.join(depth_path, folder_name_1[x]) + "/" for x in range(folder_name_1.shape[0])])
    path_1 = np.array([path + x + "/" for x in folder_name_1])
    save_path_0 = save_path + "megadepth_parameters/"
    Create_files(save_path_0)
    for i in np.array(range(depth_path_1.shape[0] - depth_path_1.shape[0] + 1)) + depth_path_1.shape[0] - 1:
        save_path_1 = save_path_0 + folder_name_1[i] + "/"
        print(save_path_1)
        Create_files(save_path_1)
        folder_name_2 = np.sort(np.array(os.listdir(depth_path_1[i])))
        depth_path_2 = np.array(
            [os.path.join(depth_path_1[i], folder_name_2[x]) + "/" for x in range(folder_name_2.shape[0])])
        for j in range(depth_path_2.shape[0]):
            save_path_2 = save_path_1 + folder_name_2[j] + "/"
            Create_files(save_path_2)
            depth_path_3 = depth_path_2[j] + "depths/"
            path_2 = path_1[i] + "sparse/manhattan/" + str(j) + "/"
            if not os.path.exists(path_2):
                path_2 = path_1[i] + "sparse/manhattan/" + str(j + 1) + "/"
            Megadepth_preprocess(path_2, depth_path_3, save_path_2)


def Auto_depth_check_script(depth_path, param_path, shape=np.array([640, 480]), patch_size=16):
    folder_name_1 = np.sort(np.array(os.listdir(depth_path)))
    depth_path_1 = np.array([os.path.join(depth_path, folder_name_1[x]) + "/" for x in range(folder_name_1.shape[0])])
    param_path_0 = param_path + "megadepth_parameters/"
    save_path_0 = param_path + "megadepth_accuracy_reverse_" + str(patch_size * 2) + "/"
    Create_files(save_path_0)
    for i in np.array(range(depth_path_1.shape[0] - depth_path_1.shape[0] + 1)) + 34:
        if os.path.exists(param_path_0 + folder_name_1[i]):
            param_path_1 = param_path_0 + folder_name_1[i] + "/"
            save_path_1 = save_path_0 + folder_name_1[i] + "/"
            Create_files(save_path_1)
            folder_name_2 = np.sort(np.array(os.listdir(depth_path_1[i])))
            depth_path_2 = np.array([os.path.join(depth_path_1[i], folder_name_2[x]) + "/" for x in range(folder_name_2.shape[0])])
            for j in np.array(range(depth_path_2.shape[0])):
                if os.path.exists(param_path_0 + folder_name_1[i] + "/" + folder_name_2[j]):
                    print(param_path_0 + folder_name_1[i] + "/" + folder_name_2[j])
                    param_path_2 = param_path_1 + folder_name_2[j] + "/"
                    img_path_3 = depth_path_2[j] + "imgs/"
                    save_path_2 = save_path_1 + folder_name_2[j] + "/"
                    Create_files(save_path_2)
                    depth_path_3 = depth_path_2[j] + "depths/"
                    # Depth_check_process(depth_path_3, param_path_2, save_path_2, img_path_3, patch_size=patch_size)
                    Label_Compute_process(depth_path_3, param_path_2, save_path_2, img_path_3, shape, patch_size=patch_size)

def Auto_filenumber_check_script(depth_path, param_path, patch_size=16):
    folder_name_1 = np.sort(np.array(os.listdir(depth_path)))
    depth_path_1 = np.array([os.path.join(depth_path, folder_name_1[x]) + "/" for x in range(folder_name_1.shape[0])])
    param_path_0 = param_path + "megadepth_parameters/"
    save_path_0 = param_path + "megadepth_accuracy_reverse_" + str(patch_size * 2) + "/"
    Create_files(save_path_0)
    for i in np.array(range(depth_path_1.shape[0])):
        param_path_1 = param_path_0 + folder_name_1[i] + "/"
        save_path_1 = save_path_0 + folder_name_1[i] + "/"
        folder_name_2 = np.sort(np.array(os.listdir(depth_path_1[i])))
        depth_path_2 = np.array([os.path.join(depth_path_1[i], folder_name_2[x]) + "/" for x in range(folder_name_2.shape[0])])
        for j in range(depth_path_2.shape[0]):
            save_path_2 = save_path_1 + folder_name_2[j] + "/"
            folder_name_3 = np.sort(np.array(os.listdir(save_path_2)))
            param_path_2 = param_path_1 + folder_name_2[j] + "/pair_list.txt"
            num_expect = Get_pairs(param_path_2).shape[0]
            num_real = folder_name_3.shape[0]
            if num_real < num_expect:
                print(save_path_2)




def Depth_check_process(depth_path, path, savepath, imgpath=None, shape=np.array([640, 480]), patch_size=4):
    param_path = path + "img_cam.txt"
    pair_path = path + "pair_list.txt"
    save_path = savepath + "accuracy"
    (name_list, size_list, K_list, P_list) = Get_params(param_path)
    pair_list = Get_pairs(pair_path)
    accuracy_list = []
    success_nums = []
    for i in range(pair_list.shape[0]):
        name1 = pair_list[i][0].split('.')[0]
        name2 = pair_list[i][1].split('.')[0]
        p = name_list[pair_list[i][0]]
        q = name_list[pair_list[i][1]]
        P = Create_P_resize(K_list[p], K_list[q], P_list[p], P_list[q], size_list[p], size_list[q])
        depth1 = h5py.File(depth_path + name1 + ".h5", "r")['depth']
        depth2 = h5py.File(depth_path + name2 + ".h5", "r")['depth']
        depth1 = Resize_depth(depth1, shape)
        depth2 = Resize_depth(depth2, shape)
        (accuracy, success_num) = Compute_label_accuracy(depth1, depth2, P, depth1.shape, patch_size)
        accuracy_list.append(accuracy)
        success_nums.append(success_num)
        if i % 30 == 29:
            print(np.mean(accuracy_list), np.mean(success_nums))
            accuracy_list.clear()
    np.save(save_path, np.array(accuracy_list))

def Label_Compute_thread(correspondense_list, depth_path, pair_list, name_list, K_list, P_list, size_list, shape, i, save_path, patch_size=16, num=1000):
    print(i)
    row_num = shape[0] // patch_size // 2
    col_num = shape[1] // patch_size // 2
    num_input = min(num, pair_list.shape[0] - i * num)
    depth1_whole = np.zeros((num_input, shape[1], shape[0]))
    depth2_whole = np.zeros((num_input, shape[1], shape[0]))
    P_whole = np.zeros((num_input, 4, 4))
    f1 = []
    f2 = []
    # correspondense = np.zeros(num, row_num * col_num, 2)
    for j in range(num_input):
        sequence = i * num + j
        name1 = pair_list[sequence][0].split('.')[0]
        name2 = pair_list[sequence][1].split('.')[0]
        p = name_list[pair_list[sequence][0]]
        q = name_list[pair_list[sequence][1]]
        f1.append(K_list[p][0, 0])
        f2.append(K_list[q][0, 0])
        P_whole[j] = Create_P_resize(K_list[p], K_list[q], P_list[p], P_list[q], size_list[p], size_list[q])
        depth1 = h5py.File(depth_path + name1 + ".h5", "r")['depth']
        depth2 = h5py.File(depth_path + name2 + ".h5", "r")['depth']
        depth1_whole[j] = Resize_depth(depth1, shape)
        depth2_whole[j] = Resize_depth(depth2, shape)
    if num_input != 0:
        correspondense_list[i, 0:num_input, :, :] = Compute_label_correspondes_matrix(depth1_whole, depth2_whole, P_whole,
                                                           depth1_whole[0].shape, save_path, num_input,
                                                           patch_size, np.array(f1), np.array(f2))
    for j in range(100):
        np.save(save_path+str(i*100+j), correspondense_list[i, j])
    return

def Label_view_process(path, imgpath, shape, correspondense_list, patch_size, num):
    row_num = shape[0] // patch_size // 2
    col_num = shape[1] // patch_size // 2
    pair_path = path + "pair_list.txt"
    pair_list = Get_pairs(pair_path)
    print(pair_list.shape[0] // num + 1)
    for i in range(pair_list.shape[0] // num + 1):
        num_input = min(num, pair_list.shape[0] - i * num)
        imgs = []
        correspondense = correspondense_list[i]
        for j in range(num_input):
            sequence = i * num + j
            name1 = pair_list[sequence][0].split('.')[0]
            name2 = pair_list[sequence][1].split('.')[0]
            img1 = cv2.imread(imgpath + name1 + ".jpg")
            img2 = cv2.imread(imgpath + name2 + ".jpg")
            img1 = Resize_img(img1, shape)
            img2 = Resize_img(img2, shape)
            img3 = cv2.hconcat([img1, img2])
            imgs.append(img3)
        for p in range(num):
            for q in range(col_num):
                for t in range(row_num):
                    if correspondense[p, q * row_num + t, 0] > -0.1:
                        cv2.line(imgs[p], (int((2 * t + 1) * patch_size), int((2 * q + 1) * patch_size)),
                                 (int(correspondense[p, q * row_num + t, 0] + 640),
                                  int(correspondense[p, q * row_num + t, 1])),
                                 [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)])
            cv2.imwrite("/media/junjieni/B0AA57DDAA579F22/dataset/MegaDepth/test" + str(i*num_input+p) + ".jpg", imgs[p])

def Label_Compute_process(depth_path, path, savepath, imgpath, shape, patch_size=16, num=100, view=False):
    global correspondense
    param_path = path + "img_cam.txt"
    pair_path = path + "pair_list.txt"
    save_path = savepath
    (name_list, size_list, K_list, P_list) = Get_params(param_path)
    pair_list = Get_pairs(pair_path)
    if view:
        correspondense_list = np.load(save_path + ".npy", allow_pickle=True)
        Label_view_process(path, imgpath, shape, correspondense_list, patch_size, num)
        return
    pool = mp.Pool(4)
    thread_num = pair_list.shape[0] // num + 1
    correspondense_list = np.zeros((thread_num, num, shape[1]//patch_size//2 * shape[0]//patch_size//2, 3))
    results = [pool.apply_async(Label_Compute_thread, args=(correspondense_list, depth_path, pair_list,
                    name_list, K_list, P_list, size_list, shape, i, save_path, patch_size, num)) for i in range(thread_num)]
    results_list = [p.get() for p in results]

def Compute_label_accuracy(depth0, depth1, P, shape, patch_size=4):
    # patch_size is the distance between the point and the board of patch
    upper_bound = 1e7
    lower_bound = 1e-2
    if_success = np.zeros((shape[0] // patch_size // 2, shape[1] // patch_size // 2))
    for i in range(shape[0] // patch_size // 2):
        for j in range(shape[1] // patch_size // 2):
            x = np.zeros(2)
            y = np.zeros(2)
            x[0] = np.array((j + 0.5) * 2 * patch_size - 0.5)
            x[1] = np.array((i + 0.5) * 2 * patch_size - 0.5)
            d1_whole = depth0[(2 * i + 1) * patch_size - 1:(2 * i + 1) * patch_size,
                       (j * 2 + 1) * patch_size - 1:(2 * j + 1) * patch_size]
            d1 = np.mean(d1_whole[d1_whole > lower_bound])
            if d1 < lower_bound:
                d1 = upper_bound
            p1 = np.array([x[0] * d1, x[1] * d1, d1, 1])
            p21 = P.dot(p1)
            p21 = p21 / p21[2]
            if p21[0] > shape[1] or p21[1] > shape[0] or p21[0] < 0 or p21[1] < 0:
                if_success[i, j] = -1
                continue
            y[0] = p21[0]
            y[1] = p21[1]
            y = y.astype(int)
            d2_whole = depth1[y[1] - 1:y[1] + 1, y[0] - 1:y[0] + 1]
            d2 = np.mean(d2_whole[d2_whole > lower_bound])
            if d2 < lower_bound:
                d2 = upper_bound
            p2 = np.array([y[0] * d2, y[1] * d2, d2, 1])
            p12 = np.linalg.inv(P).dot(p2)
            p12 = p12 / p12[2]
            distance = np.abs(p1 / d1 - p12)
            if not (distance[0] > 1 or distance[1] > 1):
                if_success[i, j] = 1
    return (np.mean(if_success[if_success > -0.1]), if_success[if_success > -0.1].size)


def Compute_label_correspondes(depth0, depth1, P, shape, patch_size=4):
    # patch_size is the distance between the point and the board of patch
    upper_bound = 1e7
    lower_bound = 1e-2
    correspondense = np.zeros((shape[0] // patch_size // 2, shape[1] // patch_size // 2, 2))
    p21_test = np.zeros((shape[0] // patch_size // 2 * shape[1] // patch_size // 2, 4))
    p12_test = np.zeros((shape[0] // patch_size // 2 * shape[1] // patch_size // 2, 4))
    d1_test = np.ones((shape[0] // patch_size // 2 * shape[1] // patch_size // 2))
    for i in range(shape[0] // patch_size // 2):
        for j in range(shape[1] // patch_size // 2):
            x = np.zeros(2)
            y = np.zeros(2)
            x[0] = np.array((j + 0.5) * 2 * patch_size - 0.5)
            x[1] = np.array((i + 0.5) * 2 * patch_size - 0.5)
            d1_whole = depth0[(2 * i + 1) * patch_size - 1:(2 * i + 1) * patch_size,
                       (j * 2 + 1) * patch_size - 1:(2 * j + 1) * patch_size]
            d1 = np.mean(d1_whole[d1_whole > lower_bound])
            if d1 < lower_bound or np.isnan(d1):
                d1 = upper_bound
            d1_test[i * shape[1] // patch_size // 2 + j] = d1
            p1 = np.array([x[0] * d1, x[1] * d1, d1, 1])
            p21 = P.dot(p1)
            p21 = p21 / p21[2]
            p21_test[i * shape[1] // patch_size // 2 + j] = p21
            if p21[0] > shape[1] or p21[1] > shape[0] or p21[0] < 0 or p21[1] < 0:
                correspondense[i, j, :] = -1
                continue
            y[0] = p21[0]
            y[1] = p21[1]
            y = y.astype(int)
            d2_whole = depth1[y[1] - 1:y[1] + 1, y[0] - 1:y[0] + 1]
            d2 = np.mean(d2_whole[d2_whole > lower_bound])
            if d2 < lower_bound:
                d2 = upper_bound
            p2 = np.array([y[0] * d2, y[1] * d2, d2, 1])
            p12 = np.linalg.inv(P).dot(p2)
            p12 = p12 / p12[2]
            p12_test[i * shape[1] // patch_size // 2 + j] = p12
            distance = np.abs(p1 / d1 - p12)
            result_n = np.zeros(2)
            result_n[0] = int(p21[0])
            result_n[1] = int(p21[1])
            if not (distance[0] > patch_size / 2 or distance[1] > patch_size / 2):
                correspondense[i, j] = result_n
            else:
                correspondense[i, j, :] = -1
    return (correspondense, p21_test, p12_test, d1_test)

def Compute_label_correspondes_matrix(depth0, depth1, P, shape, save_path, num, patch_size, f0, f1):
    # patch_size is the distance between the point and the board of patch
    upper_bound = 1e7
    lower_bound = 1e-2
    row_num = shape[0] // patch_size // 2
    col_num = shape[1] // patch_size // 2
    row = np.arange(col_num) * 2 * patch_size + patch_size - 0.5
    rows = row.reshape(1, col_num).repeat(row_num, axis=0).reshape(1, row_num * col_num, 1)
    rows = rows.repeat(num, axis=0)
    col = np.arange(row_num) * 2 * patch_size + patch_size - 0.5
    cols = col.reshape(row_num, 1).repeat(col_num, axis=1).reshape(1, row_num * col_num, 1)
    cols = cols.repeat(num, axis=0)
    prefix = np.arange(num).reshape(num, 1).repeat(row_num * col_num * 4,
                                                   axis=1).reshape(num, row_num*col_num, 4)
    rows_new = rows.astype(int)
    cols_new = cols.astype(int)
    depth_input_row = np.concatenate((rows_new, rows_new, rows_new + 1, rows_new + 1), axis=2)
    depth_input_col = np.concatenate((cols_new, cols_new + 1, cols_new, cols_new + 1), axis=2)
    d0 = depth0[prefix, depth_input_col, depth_input_row]
    d0_weights = (d0 > lower_bound).astype(int)
    d0_weights[np.max(d0, axis=2) < lower_bound] = 1
    d0 = np.average(d0, weights=d0_weights, axis=2).reshape(num, col_num * row_num, 1)
    whole_num = np.sum((d0 > lower_bound).astype(int))
    if_d0 = d0 < lower_bound
    d0[d0 < lower_bound] = upper_bound
    last_ones = np.ones((num, col_num*row_num, 1))
    point_input = np.concatenate((rows * d0, cols * d0, d0, last_ones), axis=2)
    point_output = np.einsum('ijk,ipk->ipj', P, point_input)
    point_output[:, :, 0] = point_output[:, :, 0] / point_output[:, :, 2]
    point_output[:, :, 1] = point_output[:, :, 1] / point_output[:, :, 2]
    correspondense = np.zeros((point_output.shape[0], point_output.shape[1], 3))
    correspondense[:, :, 0:2] = point_output[:, :, 0:2]
    # correspondense_test, test_out, test2_out, d0_test = Compute_label_correspondes(depth0[1], depth1[1], P[1], shape, patch_size)
    if_outlier1 = np.logical_and(np.logical_or(point_output[:, :, 0] < 1, point_output[:, :, 0] >= shape[1] - 1), np.logical_not(if_d0.reshape(d0.shape[0], d0.shape[1])))
    if_outlier2 = np.logical_and(np.logical_or(point_output[:, :, 1] < 1, point_output[:, :, 1] >= shape[0] - 1), np.logical_not(if_d0.reshape(d0.shape[0], d0.shape[1])))
    correspondense[if_d0.repeat(3, axis=2)] = -1
    output_row = np.around(point_output[:, :, 0]).reshape(num, row_num * col_num, 1).astype(int)
    output_row[np.logical_or(point_output[:, :, 0] < 2, point_output[:, :, 0] >= shape[1] - 2)] = int(shape[1] / 2)
    output_col = np.around(point_output[:, :, 1]).reshape(num, row_num * col_num, 1).astype(int)
    output_col[np.logical_or(point_output[:, :, 1] < 2, point_output[:, :, 1] >= shape[0] - 2)] = int(shape[0] / 2)
    output_cols = output_col.repeat(3, axis=2)
    depth1_input_row = np.concatenate((output_row - 1, output_row, output_row + 1), axis=2).repeat(3, axis=2)
    depth1_input_col = np.concatenate((output_cols - 1, output_cols, output_cols + 1), axis=2)
    prefix = np.arange(num).reshape(num, 1).repeat(row_num * col_num * 9,
                                                   axis=1).reshape(num, row_num*col_num, 9)
    d1 = depth1[prefix, depth1_input_col, depth1_input_row]
    d1_weights = (d1 > lower_bound).astype(int)
    d1_weights[np.max(d1, axis=2) < lower_bound] = 1
    d1 = np.average(d1, weights=d1_weights, axis=2).reshape(num, col_num * row_num, 1)
    d1[d1 < lower_bound] = upper_bound
    medium = f0.reshape(f0.shape[0], 1).repeat(row_num * col_num, axis=1) / f1.reshape(f0.shape[0], 1).repeat(row_num * col_num, axis=1)\
        * d1.reshape(point_output.shape[0], -1) / d0.reshape(point_output.shape[0], -1)
    correspondense[:, :, 2] = medium
    point_input2 = np.concatenate((output_row * d1, output_col * d1, d1, last_ones), axis=2)
    point_output2 = np.einsum('ijk,ipk->ipj', np.linalg.inv(P), point_input2)
    point_output2[:, :, 1] = point_output2[:, :, 1] / point_output2[:, :, 2]
    point_output2[:, :, 0] = point_output2[:, :, 0] / point_output2[:, :, 2]
    distance_vector = np.abs(point_input / d0 - point_output2)
    distance = np.sqrt(np.power(distance_vector[:, :, 0], 2) + np.power(distance_vector[:, :, 1], 2))
    correspondense[distance.reshape(distance.shape[0], distance.shape[1], 1).repeat(3, axis=2) > 8] = -1
    correspondense[:, :, 0][if_outlier1] = -upper_bound
    correspondense[:, :, 0][if_outlier2] = -upper_bound
    fail_theory = np.sum((correspondense[:, :, 0] < -1e2).astype(int))
    true_theory = whole_num - fail_theory
    true_reality = np.sum((correspondense[:, :, 0] > -1e-2).astype(int))
    weight = true_theory / fail_theory * true_reality / true_theory
    print(true_theory, fail_theory, true_reality)
    # np.save(save_path, correspondense)
    return correspondense

def test_pair_points(points, points_list, pairs, images, n, param_path, depth0, depht1):
    path = "/media/junjieni/B0AA57DDAA579F22/dataset/MegaDepth/"
    # img0 = images[pairs[n, 0]]
    # img1 = images[pairs[n, 1]]
    depth0 = Resize_depth(depth0)
    # depth1 = Resize_depth(depht1)
    # img0_plist = []
    # img1_plist = []
    # p3d_list = []
    # for i in range(img0["keypoints"].shape[0]):
    #     result = -1
    #     for j in range(img1["keypoints"].shape[0]):
    #         if int(img1["keypoints"][j, 2]) == img0["keypoints"][i, 2]:
    #             result = int(j)
    #     if result > -0.1 and img1["keypoints"][result, 2] != -1:
    #         p3d_list.append(points[points_list[str(int(img0["keypoints"][i, 2]))]]["params"][1:4])
    #         img0_plist.append(img0["keypoints"][i, 0:2])
    #         img1_plist.append(img1["keypoints"][result, 0:2])
    # p3d_list = np.array(p3d_list)
    # img0_plist = np.array(img0_plist)
    # img1_plist = np.array(img1_plist)
    # np.save(path + "test_i0", img0_plist)
    # np.save(path + "test_i1", img1_plist)
    # np.save(path + "test_p3d", p3d_list)
    img0_plist = np.load(path + "test_i0.npy")
    img1_plist = np.load(path + "test_i1.npy")
    p3d_list = np.load(path + "test_p3d.npy")
    (name_list, size_list, K_list, P_list) = Get_params(param_path + "img_cam.txt")
    p = name_list[pairs[n][0]]
    q = name_list[pairs[n][1]]
    print(pairs[n])
    P = Create_P_resize(K_list[p], K_list[q], P_list[p], P_list[q], size_list[p], size_list[q])
    r0 = Get_resize_ratio(size_list[p], np.array([640, 480]))
    r1 = Get_resize_ratio(size_list[q], np.array([640, 480]))
    for i in range(p3d_list.shape[0]):
        x = img0_plist[i][0] * r0
        y = img0_plist[i][1] * r0
        d0 = depth0[int(y), int(x)]
        p0 = [x*d0, y*d0, d0, 1]
        p3d = [p3d_list[i][0], p3d_list[i][1], p3d_list[i][2], 1]
        p10 = P.dot(p0)
        p10 = p10 / p10[2]
        print(p3d, p10, img1_plist[i] * r1)

def Extract_merged_label(label_path, amount = 100):
    labels = np.load(label_path + "accuracy.npy")
    label_dir = label_path
    for i in range(labels.shape[0]):
        for j in range(amount):
            np.save(label_dir + str(i * amount + j), labels[i, j])

def Auto_extract_process(path):
    folder_name_1 = np.sort(np.array(os.listdir(path)))
    for i in np.array(range(folder_name_1.shape[0])):
        folder_name_2 = np.sort(np.array(os.listdir(path + folder_name_1[i])))
        for j in range(folder_name_2.shape[0]):
            Extract_merged_label(path + folder_name_1[i] + "/" + folder_name_2[j] + "/")

def Auto_resize_process(depth_path, param_path):
    folder_name_1 = np.sort(np.array(os.listdir(depth_path)))
    depth_path_1 = np.array([os.path.join(depth_path, folder_name_1[x]) + "/" for x in range(folder_name_1.shape[0])])
    save_path_0 = param_path + "megadepth_resized" + "/"
    print(save_path_0)
    Create_files(save_path_0)
    for i in np.array(range(depth_path_1.shape[0])):
        print(i)
        save_path_1 = save_path_0 + folder_name_1[i] + "/"
        print(save_path_1)
        Create_files(save_path_1)
        folder_name_2 = np.sort(np.array(os.listdir(depth_path_1[i])))
        depth_path_2 = np.array([os.path.join(depth_path_1[i], folder_name_2[x]) + "/" for x in range(folder_name_2.shape[0])])
        for j in range(depth_path_2.shape[0]):
            img_path_3 = depth_path_2[j] + "imgs/"
            folder_name_3 = np.sort(np.array(os.listdir(img_path_3)))
            img_path_3 = np.array([os.path.join(img_path_3, folder_name_3[x]) for x in range(folder_name_3.shape[0])])
            save_path_2 = save_path_1 + folder_name_2[j] + "/"
            print(save_path_2)
            Create_files(save_path_2)
            for k in range(folder_name_3.shape[0]):
                img = cv2.imread(img_path_3[k])
                img = Resize_img(img, np.array([640, 480]))
                cv2.imwrite(save_path_2 + folder_name_3[k], img)

def Auto_resize_kitti(depth_path, param_path):
    save_path_0 = param_path + "kitti_00_resized_h4shrink" + "/"
    Create_files(save_path_0)
    img_path = depth_path + "image_2/"
    folder_name_1 = np.sort(np.array(os.listdir(img_path)))
    for k in range(folder_name_1.shape[0]):
        print(img_path + folder_name_1[k])
        img = cv2.imread(img_path + folder_name_1[k])
        img = cv2.resize(img, (310, 376))
        img = Resize_img(img, np.array([640, 480]))
        cv2.imwrite(save_path_0 + folder_name_1[k], img)


def pre_process(img, save_path, device):
    model = feature_extractor()
    pretrained_dict = models.vgg16(pretrained=True).state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    img = torch.from_numpy(img).to(device).reshape(1, img.shape[0], img.shape[1], img.shape[2])\
        .permute(0, 3, 1, 2).float()
    feature = model(img)
    np.save(save_path, feature.cpu().numpy())

def Auto_feature_process(depth_path, param_path, device):
    folder_name_1 = np.sort(np.array(os.listdir(depth_path)))
    depth_path_1 = np.array([os.path.join(depth_path, folder_name_1[x]) + "/" for x in range(folder_name_1.shape[0])])
    save_path_0 = param_path + "megadepth_features" + "/"
    print(save_path_0)
    Create_files(save_path_0)
    for i in np.array(range(depth_path_1.shape[0] - 171)) + 171:
        print(i)
        save_path_1 = save_path_0 + folder_name_1[i] + "/"
        print(save_path_1)
        Create_files(save_path_1)
        folder_name_2 = np.sort(np.array(os.listdir(depth_path_1[i])))
        depth_path_2 = np.array([os.path.join(depth_path_1[i], folder_name_2[x]) + "/" for x in range(folder_name_2.shape[0])])
        for j in range(depth_path_2.shape[0]):
            img_path_3 = depth_path_2[j]
            folder_name_3 = np.sort(np.array(os.listdir(img_path_3)))
            img_path_3 = np.array([os.path.join(img_path_3, folder_name_3[x]) for x in range(folder_name_3.shape[0])])
            save_path_2 = save_path_1 + folder_name_2[j] + "/"
            print(save_path_2)
            Create_files(save_path_2)
            for k in range(folder_name_3.shape[0]):
                img = cv2.imread(img_path_3[k])
                name = folder_name_3[k].split('.')[0]
                pre_process(img, save_path_2 + name, device)


def epipolar_progress(pairs, K_list, P_list, name_list, size_list, save_path):
    epipolar_list=[]
    for i in range(pairs.shape[0]):
        name_l = pairs[i][0]
        name_r = pairs[i][1]
        id_l = name_list[name_l]
        id_r = name_list[name_r]
        epipolar = compute_F(P_list[id_l], P_list[id_r], K_list[id_l], K_list[id_r], size_list[id_l], size_list[id_r])
        epipolar_list.append(epipolar)
    np.save(save_path + "epipolar", np.array(epipolar_list))

def Auto_epipolar_script(param_path):
    param_path_0 = param_path + "megadepth_parameters/"
    save_path_0 = param_path + "megadepth_parameters/"
    Create_files(save_path_0)
    folder_name_1 = np.sort(np.array(os.listdir(param_path_0)))
    param_path_1 = np.array([os.path.join(param_path_0, folder_name_1[x]) + "/" for x in range(folder_name_1.shape[0])])
    for i in np.array(range(param_path_1.shape[0])):
        if os.path.exists(param_path_0 + folder_name_1[i]):
            save_path_1 = save_path_0 + folder_name_1[i] + "/"
            print(save_path_1)
            Create_files(save_path_1)
            folder_name_2 = np.sort(np.array(os.listdir(param_path_1[i])))
            param_path_2 = np.array([os.path.join(param_path_1[i], folder_name_2[x]) + "/" for x in range(folder_name_2.shape[0])])
            for j in np.array(range(param_path_2.shape[0])):
                if os.path.exists(param_path_0 + folder_name_1[i] + "/" + folder_name_2[j]):
                    param_path_2 = param_path_1[i] + folder_name_2[j] + "/pair_list.txt"
                    param_path_3 = param_path_1[i] + folder_name_2[j] + "/img_cam.txt"
                    pairs = Get_pairs(param_path_2)
                    (name_list, size_list, K_list, P_list) = Get_params(param_path_3)
                    save_path_2 = save_path_1 + folder_name_2[j] + "/"
                    Create_files(save_path_2)
                    epipolar_progress(pairs, K_list, P_list, name_list, size_list, save_path_2)


path = "/media/junjieni/B0AA57DDAA579F22/dataset/MegaDepth/MegaDepth_v1_SfM/"
depth_path = "/media/junjieni/8E6E5E5E6E5E3EE1/dataset/MegaDepth_v1/"
save_path = "/media/junjieni/B0AA57DDAA579F22/dataset/MegaDepth/"
save_path_eazier = "/media/junjieni/8E6E5E5E6E5E3EE1/dataset/easier_megadepth/"
caps_path = "/media/junjieni/B0AA57DDAA579F22/dataset/megadepth/"
resized_path = "/media/junjieni/B0AA57DDAA579F22/dataset/MegaDepth/megadepth_resized/"
kitti_00_path = "/media/junjieni/8E6E5E5E6E5E3EE1/dataset/sequences/00/"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
Auto_epipolar_script(save_path_eazier)
# Auto_process_script(path, depth_path, save_path_eazier)
# Auto_depth_check_script(depth_path, save_path_eazier)
# Auto_filenumber_check_script(depth_path, save_path)
# Auto_extract_process(save_path + "megadepth_accuracy_32/")
# Auto_resize_process(depth_path, save_path)
# Auto_resize_kitti(kitti_00_path, save_path)
# Auto_feature_process(resized_path, save_path, device)