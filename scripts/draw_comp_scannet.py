from ast import comprehension
from operator import index
from traceback import print_stack
from turtle import left, right
import numpy as np
import cv2
import tqdm


result_path = '/home/eugenelee/Downloads/SAPT/code/scannet_comp'
data_path = ''

def draw_rectangle(img, pt, size, colors):
    for i in range(len(pt)):
        nw = (int(pt[i][1]), int(pt[i][0]))
        se = (int(pt[i][1]+size), int(pt[i][0]+size))
        cv2.rectangle(img, nw, se, colors[i].tolist(), -1)
    return img

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

def check_valid(pair):
    flag = True
    if pair == -1:
        flag = False
    kp1 = pair["matches_l"].cpu().numpy()
    if kp1.shape[0] < 15:
        flag = False
    return flag

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

def draw_method(results, dist, draw_circle, s, method, index_list=None, return_text=False, blend_weight=1.0):
    if index_list is not None:
        _list = index_list
    else:
        _list = list(range(300))
    ret = []
    ret_text = []
    white_index = [3313, 3420]
    for i in tqdm.tqdm(_list):
        pair = results[i]
        # if not check_valid(pair):
        #     continue
        # ep = np.abs(pair["epipolar_distance"])

        left_name, right_name = pair["left_path"][0], pair["right_path"][0]
        left_name, right_name, _ =left_name.split(".jpg")
        left_name = left_name + ".jpg"
        right_name = right_name + ".jpg"
        left_img = cv2.imread(left_name)
        right_img = cv2.imread(right_name)
        left_img = Resize_img(left_img)
        right_img = Resize_img(right_img)
        lh, lw, rh, rw = left_img.shape[0], left_img.shape[1], right_img.shape[0], right_img.shape[1]

        kp1 = pair["matches_l"].cpu().numpy()
        kp2 = pair["matches_r"].cpu().numpy()
        mask1 = np.logical_and.reduce(np.array((kp1[:,1]>=0, kp1[:,1]<lw, kp1[:,0]>=0, kp1[:,0]<lh)))
        mask2 = np.logical_and.reduce(np.array((kp2[:,1]>=0, kp2[:,1]<rw, kp2[:,0]>=0, kp2[:,0]<rh)))
        # print(f'{np.count_nonzero(mask1)}/{mask1.size}')
        # print(f'{np.count_nonzero(mask2)}/{mask2.size}')

        if check_valid(pair):
            ep = np.abs(pair["epipolar_distance"])
        else:
            num = kp1.shape[0]
            ep = np.ones([num])

        mask = np.logical_and.reduce(np.array((mask1, mask2, ep<dist)))
        # print(f'{np.count_nonzero(mask)}/{mask.size}')
        kp1 = kp1[mask]
        kp2 = kp2[mask]

        color = kp_color(kp1[:,1], kp1[:,0], (lh, lw))
        left_img = draw_func(left_img, kp1, color, size=s, draw_circle=draw_circle, blend_weight=blend_weight)
        right_img = draw_func(right_img, kp2, color, size=s, draw_circle=draw_circle, blend_weight=blend_weight)

        max_height = max(lh, rh)
        pad_width = 5
        vis = np.ones([max_height, lw+rw+pad_width, 3], dtype=left_img.dtype) * 255
        vis[:lh, :lw, :] = left_img
        vis[:rh, lw+pad_width:lw+pad_width+rw, :] = right_img

        if check_valid(pair):
            r_err, t_err = pair['R_error'], pair['T_error']
        else:
            r_err, t_err = 1e4, 1e4
        print_str = [f'Method: {config[method]["print_name"]}',
                    f'Inliers: {np.count_nonzero(mask)}/{mask.size}',
                    f'Error_Rot: {r_err:.1f}',
                    f'Error_T: {t_err:.1f}']
        if return_text:
            ret_text.append(print_str)
            pass
        else:
            sum_height = 0
            pad_height = 8
            pad_left = 5
            text_color = (0,0,0) if i not in white_index else (255,255,255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            thickness = 1
            for _str in print_str:
                # import ipdb; ipdb.set_trace()
                (tw, th), _ = cv2.getTextSize(_str, font, fontScale, thickness)
                cv2.putText(vis, _str, (pad_left,sum_height+pad_height+th), font, fontScale, text_color, thickness, cv2.LINE_AA, bottomLeftOrigin=False)
                sum_height += th + pad_height

        # cv2.imshow('vis', vis)
        # cv2.waitKey(0)

        ret.append(vis)
        cv2.imwrite(f"/home/eugenelee/Downloads/SAPT/code/scannet_comp/{method}/{i:04d}.jpg", vis)

    if return_text: return ret, ret_text
    else: return ret


def draw_all_method():
    draw_circle=True
    dist = 5
    # method='loftr'
    # for method in ['superglue', 'loftr', 'ours', 'aspanformer']:
    for method in ['loftr', 'ours']:
        s = config[method]['circle_size']
        results = list(np.load(f"{result_path}/scannet_{method}.npy", allow_pickle=True))
        draw_method(results, dist, draw_circle, s, method)

def draw_comp():
    dist = 5
    # method='loftr'
    index_list = [20, 118, 158, 200, 210, 262, 264, 285, 305, 329, 444, 445, 448, 463, 485, 504, 565, 589, 590, 597, 599, 602, 607, 694, 701, 779, 788, 809, 829, 856, 892, 902, 917, 925, 963, 984, 985, 998, 1005, 1059, 1209, 1292, 1368, 1373, 1376, 1417, 1489, 1492, 1497]
    # index_list = [779, 925]
    # index_list = [3420]
    comp_results = []
    text_list = []
    pad_width = 15
    for method in ['superglue', 'loftr', 'ours']:
        s = config[method]['circle_size']
        results = list(np.load(f"/home/eugenelee/Downloads/SAPT/code/scannet_comp/scannet_{method}.npy", allow_pickle=True))
        ret, text = draw_method(results, dist, config[method]['draw_circle'], s, method, index_list,
                                return_text=True, blend_weight=config[method]['blend_weight'])
        comp_results.append(ret)
        text_list.append(text)
    for j in range(len(comp_results[0])):
        _vis = []
        for i in range(len(comp_results)):
            if i > 0:
                pad_img = np.ones([comp_results[i][j].shape[0], pad_width, 3], dtype=comp_results[i][j].dtype) * 255
                _vis.append(pad_img)
            _vis.append(comp_results[i][j])
        # frow = np.concatenate(_vis[0:2], axis=1)
        # srow = np.concatenate(_vis[2:4], axis=1)
        # vis = np.concatenate([frow, srow], axis=0)
        vis = np.concatenate(_vis, axis=1)
        aspect_ratio = vis.shape[0] / vis.shape[1]
        # th = 1200
        # tw = int(th / aspect_ratio)
        tw = 4000
        th = int(tw * aspect_ratio)
        vis = cv2.resize(vis, (tw, th), interpolation=cv2.INTER_AREA)

        pad_height = 12
        pad_left = 5
        text_color = (0,0,0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        thickness = 1
        stride_width = int((tw-pad_width*(len(comp_results)-1))/len(text_list))
        for i, text in enumerate(text_list):
            sum_height = 5
            for k, _str in enumerate(text[j]):
                (tw, th), _ = cv2.getTextSize(_str, font, fontScale, thickness)
                nw = (i*stride_width+i*pad_width,sum_height)
                se = (nw[0]+pad_left*2+tw, nw[1]+th+pad_height)
                if k == 0: cv2.rectangle(vis, (nw[0],0), (se[0],nw[1]), (255,255,255), -1)
                cv2.rectangle(vis, nw, se, (255,255,255), -1)
                cv2.putText(vis, _str, (nw[0]+pad_left, nw[1]+th+int(pad_height/2)), font, fontScale, text_color, thickness, cv2.LINE_AA, bottomLeftOrigin=False)
                sum_height += th + pad_height

        # cv2.imshow('', vis)
        # cv2.waitKey(0)
        cv2.imwrite(f'{result_path}/final/{index_list[j]:04d}.png', vis)


def count_good_case():
    r_gap, t_gap = 1.0, 1.0
    r_ratio, t_ratio = 5.0, 5.0
    method_results = []
    ret = []
    for method in ['superglue', 'loftr', 'ours']:
        results = list(np.load(f"{result_path}/scannet_{method}.npy", allow_pickle=True))
        method_results.append(results)
    for i in tqdm.tqdm(range(1500)):
        best_r_err, best_t_err = 1e4, 1e4
        for j in range(len(method_results)-1):
            pair = method_results[j][i]
            if not check_valid(pair):
                continue
            r_err, t_err = pair['R_error'], pair['T_error']
            best_r_err = min(best_r_err, r_err)
            best_t_err = min(best_t_err, t_err)
        pair = method_results[-1][i]
        if not check_valid(pair):
            continue
        r_err, t_err = pair['R_error'], pair['T_error']
        cond1 = (r_err + r_gap) < best_r_err and (t_err + t_gap) < best_t_err
        cond2 = (r_err * r_ratio) < best_r_err and (t_err * t_ratio) < best_t_err
        if cond1 and cond2:
            ret.append(i)
            print(f'{i:04d}, ours: [{r_err:.1f}, {t_err:.1f}], others: [{best_r_err:.1f}, {best_t_err:.1f}]')
    print(ret)



def main():
    # draw_all_method()
    # count_good_case()
    draw_comp()


if __name__ == '__main__':
    main()