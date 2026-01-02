from crypt import methods
from tokenize import endpats
import numpy as np
import cv2
import tqdm
from colmap_io import read_model
from scipy.spatial.transform.rotation import Rotation as Rot
import kornia
import torch

def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))

def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))

def get_pose_error(_kpts0, _kpts1, K0, K1, pose, threshold=1., conf=0.99999):
    if len(_kpts0) < 5:
        return np.nan, np.nan
    K0, K1 = K0.numpy(), K1.numpy()
    pose = pose.numpy()
    kpts0 = _kpts0[...,::-1].copy()
    kpts1 = _kpts1[...,::-1].copy()
    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = threshold / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    assert E is not None

    ret = None
    best_num_inliers = 0
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    if n < 1:
        return np.nan, np.nan
    R_gt = pose[:3, :3]
    t_gt = pose[:3, 3]
    R_est, t_est, inliers = ret
    error_t = angle_error_vec(t_est, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R_est, R_gt)

    return error_R, error_t


def read_sfm_result(data_path):
    cameras, images, points3D = read_model(path=data_path, ext='.bin')
    key = sorted(images.keys())
    data = []
    for i in range(len(key)):
        k = cameras[1].params
        k = [e/1920*1600 for e in k]
        K = np.array([
            [k[0], 0,    k[1]],
            [0,    k[0], k[2]],
            [0,    0,    1],
        ])
        qcw = images[key[i]].qvec
        tcw = images[key[i]].tvec
        rcw = Rot.from_quat(qcw[1:4].tolist()+qcw[0:1].tolist()).as_matrix()
        data.append([torch.from_numpy(K), torch.from_numpy(rcw), torch.from_numpy(tcw[:,None])])
    return data

def draw_kp(img, kps, colors):
    for i, kp in enumerate(kps):
        img = cv2.circle(img, (int(kp[1]), int(kp[0])), 1, colors[i].tolist(), -1)
    return img

def draw_matches(img, kps1, kps2):
    for i, kp in enumerate(kps1):
        cv2.line(img, (int(kps1[i][1]), int(kps1[i][0])), (int(kps2[i][1]), int(kps2[i][0])), (0,255,0), 1) 
    return img

# conver R,t from world-to-camera to camera-to-world
def convert_pose(R, t):
    R = R.T
    t = -R @ t
    return R, t

def compute_gt_pose(sfm_data_0, sfm_data_i):
    K1, R1, t1 = sfm_data_0
    K2, R2, t2 = sfm_data_i
    R1, t1 = convert_pose(R1, t1)
    R2, t2 = convert_pose(R2, t2)
    R = R1.T @ R2
    t = R1.T @ (t2 - t1)
    pose = torch.eye(4)
    pose[:3, :3] = R
    pose[:3, 3:4] = t

    return pose


def compute_epipolar_distance(kp1, kp2, sfm_data_0, sfm_data_i):
    K1, R1, t1 = sfm_data_0
    K2, R2, t2 = sfm_data_i
    kp1 = torch.from_numpy(kp1)
    kp2 = torch.from_numpy(kp2)
    E = kornia.geometry.epipolar.essential_from_Rt(R1, t1, R2, t2)
    res = kornia.geometry.epipolar.decompose_essential_matrix(E)
    F = kornia.geometry.epipolar.fundamental_from_essential(E, K1, K2)
    kp1 = kp1[:, [1,0]]
    kp2 = kp2[:, [1,0]]
    ep = kornia.geometry.epipolar.right_to_left_epipolar_distance(kp1[None], kp2[None], F[None].float())
    return ep[0]


def _resize_img(depth, shape=np.array([640, 480])):
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



def draw_rectangle(img, pt, size, colors):
    for i in range(len(pt)):
        nw = (int(pt[i][1]), int(pt[i][0]))
        se = (int(pt[i][1]+size), int(pt[i][0]+size))
        cv2.rectangle(img, nw, se, colors[i].tolist(), -1)
    return img


def Resize_img(left):
    size = 1600
    h_l, w_l = left.shape[:2]
    max_shape_l = max(h_l, w_l)
    size_l = size / max_shape_l
    left = _resize_img(left, np.array([int(w_l * size_l), int(h_l * size_l)]))
    h_l, w_l = left.shape[:2]
    h_l2 = h_l // 32 * 32 + (1 - int(h_l % 32 == 0)) * 32
    w_l2 = w_l // 32 * 32 + (1 - int(w_l % 32 == 0)) * 32

    max_width = w_l2
    max_height = h_l2
    left = cv2.copyMakeBorder(left, 0, max_height - h_l, 0, max_width - w_l, cv2.BORDER_CONSTANT, None, 0)
    return left



def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image

def coord_trans(u, v):
    rad = np.sqrt(np.square(u) + np.square(v))
    u /= (rad+1e-3)
    v /= (rad+1e-3)
    return u, v

def kp_color(u, v, resolution):
    h, w = resolution
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    xx, yy = coord_trans(xx, yy)
    vis = flow_uv_to_colors(xx, yy)

    color = vis[v.astype(np.int), u.astype(np.int)]
    return color

def draw_func(_img, kps, colors, size=1, draw_circle=True, blend_weight=1.0):
    img = _img.copy()
    if draw_circle:
        for i, kp in enumerate(kps):
            img = cv2.circle(img, (int(kp[1]), int(kp[0])), size, colors[i].tolist(), -1)
    else:
        # img[kps[:,0].astype(np.int), kps[:,1].astype(np.int)] = colors
        img = draw_rectangle(img, kps, size, colors)
    img = img*blend_weight + _img*(1-blend_weight)
    return img.astype(np.uint8)

config = {
    'superglue': {
        'blend_weight': 1.0,
        'draw_circle': True,
        'circle_size': 4,
        'print_name': 'SuperGlue',
    },
    'loftr': {
        'blend_weight': 1.0,
        'draw_circle': True,
        'circle_size': 4,
        'print_name': 'LoFTR',
    },
    'aspanformer': {
        'blend_weight': 1.0,
        'draw_circle': True,
        'circle_size': 5,
        'print_name': 'ASpanFormer',
    },
    'ours': {
        'blend_weight': 0.80,
        'draw_circle': False,
        'circle_size': 2,
        'print_name': 'Ours',
    },
}

def draw_method(results, data_path, draw_circle, s, method, return_text=False, blend_weight=1.0):
    ret = []
    ret_text = []
    sfm_data = read_sfm_result(f'{data_path}/model')
    _list = range(len(sfm_data) - 1)
    ct = 0
    for i in tqdm.tqdm(_list):
        ct += 1
        # if ct < 50: continue
        pair = results[i]
        left_name, right_name = pair["left_path"][0], pair["right_path"][0]
        left_name = f'{data_path}/{left_name}'
        right_name = f'{data_path}/{right_name}'
        left_img = cv2.imread(left_name)
        right_img = cv2.imread(right_name)
        left_img = Resize_img(left_img)
        right_img = Resize_img(right_img)
        lh, lw, rh, rw = left_img.shape[0], left_img.shape[1], right_img.shape[0], right_img.shape[1]

        kp1 = pair["matches_l"]
        kp2 = pair["matches_r"]
        mask1 = np.logical_and.reduce(np.array((kp1[:,1]>=0, kp1[:,1]<lw, kp1[:,0]>=0, kp1[:,0]<lh)))
        mask2 = np.logical_and.reduce(np.array((kp2[:,1]>=0, kp2[:,1]<rw, kp2[:,0]>=0, kp2[:,0]<rh)))
        # print(f'{np.count_nonzero(mask1)}/{mask1.size}')
        # print(f'{np.count_nonzero(mask2)}/{mask2.size}')

        mask = np.logical_and.reduce(np.array((mask1, mask2)))
        # print(f'{np.count_nonzero(mask)}/{mask.size}')
        kp1 = kp1[mask]
        kp2 = kp2[mask]

        ep = compute_epipolar_distance(kp1, kp2, sfm_data[0], sfm_data[i+1])

        color = kp_color(kp1[:,1], kp1[:,0], (lh, lw))

        dist = 0.5
        inlier_kp1 = kp1[ep < dist]
        inlier_kp2 = kp2[ep < dist]
        color = color[ep < dist]

        gt_pose = compute_gt_pose(sfm_data[i+1], sfm_data[0])
        err_R, err_t = get_pose_error(kp1, kp2, sfm_data[0][0], sfm_data[i+1][0], gt_pose)

        pad_width = 5
        assert lh == rh
        # max_height = max(lh, rh)
        # vis = np.ones([max_height, lw+rw+pad_width, 3], dtype=left_img.dtype) * 255
        # vis[:lh, :lw, :] = left_img
        # vis[:rh, lw+pad_width:lw+pad_width+rw, :] = right_img

        zero_image = np.zeros([lh, pad_width, 3])
        vis = np.concatenate([left_img, zero_image, right_img], axis=1)

        # left_img = draw_func(left_img, inlier_kp1, color, size=s, draw_circle=draw_circle, blend_weight=blend_weight)
        # right_img = draw_func(right_img, inlier_kp2, color, size=s, draw_circle=draw_circle, blend_weight=blend_weight)
        # vis[:, :lw] = left_img
        # vis[:, lw+pad_width:] = right_img

        left_img = draw_kp(left_img, inlier_kp1, color)
        right_img = draw_kp(right_img, inlier_kp2, color)
        inlier_kp2[:,1] += lw + pad_width
        draw_matches(vis, inlier_kp1, inlier_kp2)

        print_str = [f'Method: {config[method]["print_name"]}',
                    # f'Matches: {len(inlier_kp1):.1e}/{len(kp1):.1e}']
                    f'Matches: {len(inlier_kp1)}',
                    f'Error_Rot: {err_R:.2f}',
                    f'Error_T: {err_t:.2f}',
                    ]
        if return_text:
            ret_text.append(print_str)
            pass
        else:
            pad_height = 30
            sum_height = 10
            pad_left = 5
            text_color = (0,0,0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 2.5
            thickness = 1
            for k, _str in enumerate(print_str):
                # import ipdb; ipdb.set_trace()
                (tw, th), _ = cv2.getTextSize(_str, font, fontScale, thickness)
                nw = (0,sum_height)
                se = (nw[0]+pad_left*2+tw, nw[1]+th+pad_height)
                if k == 0: cv2.rectangle(vis, (nw[0],0), (se[0],nw[1]), (255,255,255), -1)
                cv2.rectangle(vis, nw, se, (255,255,255), -1)
                cv2.putText(vis, _str, (nw[0]+pad_left,nw[1]+th+int(pad_height/2)), font, fontScale, text_color, thickness, cv2.LINE_AA, bottomLeftOrigin=False)
                sum_height += th + pad_height

        # cv2.imshow('vis', vis)
        # cv2.waitKey(0)
        # cv2.imwrite(f"/home/eugenelee/Downloads/SAPT/scale/204/tmp.png", vis)

        ret.append(vis)
        # cv2.imwrite(f"/home/eugenelee/Downloads/SAPT/code/scannet_comp/{method}/{i:04d}.jpg", vis)

    if return_text: return ret, ret_text
    else: return ret


def draw_comp():
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    params = parser.parse_args()

    comp_results = []
    text_list = []
    pad_width = 15
    data_path = params.data_folder
    frames = []
    # ideal_order = ['superglue', 'loftr', 'aspanformer', 'ours']
    ideal_order = ['loftr', 'ours']
    method_list = sorted([e[:-4] for e in os.listdir(f'{params.data_folder}') if e.endswith('npy')])
    nlist = []
    for e in ideal_order:
        if e in method_list: nlist.append(e)
    method_list = nlist
    for method in method_list:
    # for method in ['loftr', 'ours']:
        s = config[method]['circle_size']
        results = list(np.load(f"{data_path}/{method}.npy", allow_pickle=True))
        ret = draw_method(results, data_path, config[method]['draw_circle'], s, method,
                                return_text=False, blend_weight=config[method]['blend_weight'])
        comp_results.append(ret)

        os.system(f'mkdir -p {data_path}/{method}')
        for i, r in enumerate(ret):
            aspect_ratio = r.shape[0] / r.shape[1]
            tw = 1600
            th = int(tw * aspect_ratio)
            r = cv2.resize(r, (tw, th), interpolation=cv2.INTER_AREA)
            cv2.imwrite(f'{data_path}/{method}/{i:04d}.png', r)

    # for j in range(len(comp_results[0])):
    #     _vis = []
    #     for i in range(len(comp_results)):
    #         if i > 0:
    #             pad_img = np.ones([pad_width, comp_results[i][j].shape[1], 3], dtype=comp_results[i][j].dtype) * 255
    #             _vis.append(pad_img)
    #         _vis.append(comp_results[i][j])
    #     # frow = np.concatenate(_vis[0:2], axis=1)
    #     # srow = np.concatenate(_vis[2:4], axis=1)
    #     # vis = np.concatenate([frow, srow], axis=0)
    #     vis = np.concatenate(_vis, axis=0)
    #     aspect_ratio = vis.shape[0] / vis.shape[1]
    #     # th = 1200
    #     # tw = int(th / aspect_ratio)
    #     tw = 1000
    #     th = int(tw * aspect_ratio)
    #     vis = cv2.resize(vis, (tw, th), interpolation=cv2.INTER_AREA)

    #     # cv2.imshow('', vis)
    #     # cv2.waitKey(0)
    #     cv2.imwrite(f'{data_path}/comp/{j:04d}.png', vis)
    #     frames.append(vis)

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    # video = cv2.VideoWriter(f'{data_path}/comp.mp4', fourcc, 1, (vis.shape[1], vis.shape[0]))
    # for i in range(0, len(frames)):
    #     img = frames[i]
    #     img = img.astype(np.uint8)
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     video.write(img)

    # video.release()
    # cv2.destroyAllWindows()

def triangulation(kp1, kp2, sfm_data_0, sfm_data_i):
    K1, R1, t1 = sfm_data_0
    K2, R2, t2 = sfm_data_i
    kp1 = torch.from_numpy(kp1)
    kp2 = torch.from_numpy(kp2)

    kp1 = kp1[:, [1,0]]
    kp2 = kp2[:, [1,0]]
    
    P1 = kornia.geometry.epipolar.projection_from_KRt(K1, R1, t1)
    P2 = kornia.geometry.epipolar.projection_from_KRt(K2, R2, t2)
    pt3d = kornia.geometry.epipolar.triangulate_points(P1[None], P2[None], kp1[None], kp2[None])

    return pt3d[0].numpy()

def save_two_view_recon():
    import argparse
    import open3d as o3d
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    params = parser.parse_args()

    i = 20
    method = 'ours'
    data_path = params.data_folder
    sfm_data = read_sfm_result(f'{data_path}/model')
    results = list(np.load(f"{data_path}/{method}.npy", allow_pickle=True))

    pair = results[i]
    left_name, right_name = pair["left_path"][0], pair["right_path"][0]
    left_name = f'{data_path}/{left_name}'
    right_name = f'{data_path}/{right_name}'
    left_img = cv2.imread(left_name)
    right_img = cv2.imread(right_name)
    left_img = Resize_img(left_img)
    right_img = Resize_img(right_img)
    lh, lw, rh, rw = left_img.shape[0], left_img.shape[1], right_img.shape[0], right_img.shape[1]

    kp1 = pair["matches_l"]
    kp2 = pair["matches_r"]
    mask1 = np.logical_and.reduce(np.array((kp1[:,1]>=0, kp1[:,1]<lw, kp1[:,0]>=0, kp1[:,0]<lh)))
    mask2 = np.logical_and.reduce(np.array((kp2[:,1]>=0, kp2[:,1]<rw, kp2[:,0]>=0, kp2[:,0]<rh)))

    mask = np.logical_and.reduce(np.array((mask1, mask2)))
    kp1 = kp1[mask]
    kp2 = kp2[mask]

    ep = compute_epipolar_distance(kp1, kp2, sfm_data[0], sfm_data[i+1])

    dist = 1
    kp1 = kp1[ep < dist]
    kp2 = kp2[ep < dist]

    pt3d = triangulation(kp1, kp2, sfm_data[0], sfm_data[i+1])
    mask = np.sum(np.abs(pt3d), axis=1) < 20
    pt3d = pt3d[mask]
    kp1 = kp1[mask]
    kp2 = kp2[mask]
    y, x = kp1[:,0].astype(np.int), kp1[:,1].astype(np.int)
    colors = left_img[y, x]  / 255
    colors = colors[:, [2,1,0]]
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pt3d))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(f'{data_path}/recon.ply', pcd)


def main():
    draw_comp()
    #save_two_view_recon()


if __name__ == '__main__':
    main()