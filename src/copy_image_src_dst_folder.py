import glob
from shutil import copyfile
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
 
    parser.add_argument("-src_folder", help = "Source folder path", type=str, default=None)
    parser.add_argument("-dst_folder", help = "Destination folder path", type=str, default=None)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    src_folder = args.src_folder
    dst_folder = args.dst_folder
    src_img_path_list = glob.glob(src_folder + "/*.png")
    print('Total Source Images: ', len(src_img_path_list))
    for src_img_path in tqdm(src_img_path_list):
        img_name = src_img_path.split("/")[-1]
        dst_img_path = dst_folder + "/" + img_name
        copyfile(src_img_path, dst_img_path)


if __name__ == '__main__':
    main()
