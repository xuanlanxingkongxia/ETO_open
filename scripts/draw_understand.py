from json.tool import main
from operator import irshift
from turtle import color
import numpy as np
from einops import rearrange, repeat
import cv2
import math

root = '/home/eugenelee/Downloads/SAPT/code/megadepth_understanding'

blende_weight = 0.3


def color_attenuate(color, weight):
    color = np.array(color, dtype=np.uint8)[None,None,:]
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # v = np.clip(v * weight + 50, a_min=0, a_max=255)
    s = np.ones_like(weight) * s
    h = np.ones_like(weight) * h
    v = np.ones_like(weight) * v
    hsv = cv2.merge([h,s,v]).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # import ipdb; ipdb.set_trace()
    return bgr

config = [
    {'power': [1/5,1/2]},
    {'power': [1/4.5,1/1.8]},
    {'power': [1/4.5,1/0.5]},
    {'power': [1/3,1/3]},
]

def main():
    results = np.load(f"{root}/coverage.npy", allow_pickle=True)
    sequences = [[54, 118], [88, 207], [151, 195], [32, 213]]
    ps = 32
    hps = int(ps / 2)
    for k in range(0,4):
        result = results[k]
        img_left = result["img_left_use"].clone().cpu().numpy()
        img_right = result["img_right_use"].clone().cpu().numpy()
        bound = result["bound"].cpu().numpy()
        img_left = cv2.cvtColor(img_left, cv2.COLOR_RGB2BGR)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_RGB2BGR)
        _img_left = img_left.copy()
        _img_right = img_right.copy()
        width = result["width"]
        height = result["height"]
        scores = result["scores"]
        scales = result["scales"][0]
        average_r = result["average_r"]
        sequence = result["sequence"]
        path = f"{root}/" + str(sequence)

        back_left = np.zeros_like(img_left)
        back_right = np.zeros_like(img_right)
        weight_right = np.zeros_like(img_right, dtype=np.float32)
        colors = [(20,255,255), (20,255,20)]
        line_colors = [(0,255,255), (0,255,0)]
        for i, num in enumerate(sequences[k][0:2]):
            u = num % width
            v = num // width

            # draw source patch on left image
            nw = (u*ps, v*ps)
            se = ((u+1)*ps, (v+1)*ps)
            back_left[nw[1]:se[1], nw[0]:se[0]] = colors[i]

           # draw activation on right image
            nw = (bound[0, num, 2]*ps, bound[0, num, 0]*ps)
            se = (bound[0, num, 3]*ps, bound[0, num, 1]*ps+ps)
            overlap = repeat(scores[0, num, :-1].exp(), '(h w) -> (h ph) (w pw)', h=height, w=width, ph=32, pw=32)
            overlap = (overlap * 255).cpu().numpy()
            mask = np.zeros(img_right.shape[0:2], dtype=np.bool)
            mask[nw[1]:se[1], nw[0]:se[0]] = 1
            overlap[~mask] = 0
            overlap[mask] = np.power(overlap[mask], config[k]['power'][i])
            # for i in range(10):
            #     overlap = cv2.GaussianBlur(overlap, (21,21), cv2.BORDER_DEFAULT)
            # overlap[mask] = overlap[mask] / np.max(overlap[mask]) * 255
            weight_right[mask] = overlap[mask, None] / np.max(overlap[mask])
            overlap[mask] = overlap[mask] / np.max(overlap[mask])
            overlap = color_attenuate(colors[i], overlap)
            back_right[mask] = np.clip(overlap[mask], a_min=0, a_max=255)
            # img_right = cv2.rectangle(img_right, nw, se, (0,255,0), 3) # draw cluster bbox

        _mscale = [3]
        for i, num in enumerate(sequences[k][0:2]):
            u = num % width
            v = num // width

            for mscale in _mscale:
                # draw cropping bbox on left image
                nw = (u*ps+hps - mscale*hps, v*ps+hps - mscale*hps)
                se = (u*ps+hps + mscale*hps, v*ps+hps + mscale*hps)
                img_left = cv2.rectangle(img_left, nw, se, line_colors[i], 2)
                back_left = cv2.rectangle(back_left, nw, se, line_colors[i], 2)

                nw = [int(math.fabs(i)) for i in nw]
                se = [int(math.fabs(i)) for i in se]
                cv2.imwrite(f'{path}_{num}_left.png', _img_left[nw[1]:se[1], nw[0]:se[0]])

                # draw cropping bbox on right image
                y = average_r[0, num, 0] - 0.5
                x = average_r[0, num, 1] - 0.5
                nw = (int(x*ps + hps - mscale*hps*scales[num]), int(y*ps + hps - mscale*hps*scales[num]))
                se = (int(x*ps + hps + mscale*hps*scales[num]), int(y*ps + hps + mscale*hps*scales[num]))
                img_right = cv2.rectangle(img_right, nw, se, line_colors[i], 2)
                back_right = cv2.rectangle(back_right, nw, se, line_colors[i], 2)

                nw = [int(math.fabs(i)) for i in nw]
                se = [int(math.fabs(i)) for i in se]
                cv2.imwrite(f'{path}_{num}_right.png', _img_right[nw[1]:se[1], nw[0]:se[0]])

        back_left[(np.sum(back_left,axis=2)<1)] = img_left[np.sum(back_left,axis=2)<1] * blende_weight
        weight_right[(np.sum(weight_right,axis=2)<1)] = 0.0

        img_left =  cv2.addWeighted(img_left, 0.4, back_left, 0.6, 0.0)
        # img_right =  cv2.addWeighted(img_right, 0.3, back_right, 0.7, 0.0)
        for i in range(1):
            weight_right = cv2.GaussianBlur(weight_right, (5,5), cv2.BORDER_DEFAULT)
        weight_right = 0.1 + weight_right / np.max(weight_right) * 0.7
        img_right = img_right * (1-weight_right) * blende_weight*2 + back_right * weight_right
        img_right = img_right.astype(np.uint8)
        # print(np.unique(weight_right))

        pad_img = np.ones([img_left.shape[0], 10, 3], dtype=img_left.dtype) * 255
        vis = np.concatenate([img_left, pad_img, img_right], axis=1)

        # cv2.imshow('vis', vis)
        # cv2.waitKey(0)
        cv2.imwrite(f"{path}{num}.jpg", vis)


if __name__ == '__main__':
    main()