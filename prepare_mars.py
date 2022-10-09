import os
from shutil import copyfile

# You only need to change this line to your dataset download path
download_path = './MARS/'

if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = download_path + 'pytorch'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

test_path = download_path + 'bbox_test'
query_save_path = download_path + 'pytorch/query'
gallery_save_path = download_path + 'pytorch/gallery'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)
    os.mkdir(gallery_save_path)

for root, dirs, files in os.walk(test_path):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('C')
        src_path = root + '/' + name
        dst_path = gallery_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
            dst_path = query_save_path + '/' + ID[0]  # first image is used as query image
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

os.system('rm -rf ./MARS/pytorch/query/00-1')
