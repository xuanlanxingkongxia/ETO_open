import os
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, required=True)
params = parser.parse_args()

def merge_comp():
    data_path = f'{params.data_folder}/'
    _list = sorted(os.listdir(data_path+'/comp'))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f'{data_path}/comp.mp4', fourcc, 5, (1000, 1172))

    for e in _list:
        img = cv2.imread(f'{data_path}/comp/{e}')
        img = img.astype(np.uint8)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


def merge_seperate():
    data_path = f'{params.data_folder}/'
    for method in ['superglue', 'loftr', 'aspanformer', 'ours']:
        _list = sorted(os.listdir(data_path+f'/{method}'))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(f'{data_path}/{method}.mp4', fourcc, 5, (1600, 463))

        for e in _list:
            img = cv2.imread(f'{data_path}/{method}/{e}')
            img = img.astype(np.uint8)
            video.write(img)

        cv2.destroyAllWindows()
        video.release()


def main():
    # merge_comp()
    merge_seperate()

if __name__ == '__main__':
    main()


