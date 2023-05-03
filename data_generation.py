import os
import shutil

source = "/home/moktari/Moktari/2023/Coupled_GAN/NIR2VIS_IrisVerifier/vis/iris_unrolled_VIS_JPG"
target = "/home/moktari/Moktari/2023/Coupled_GAN/NIR2VIS_IrisVerifier/VIS"

for image in os.listdir(source):
    im_path = os.path.join(source, image)
    subject = image.split("_")[0]
    des = target + "/" + subject
    os.makedirs(des, exist_ok=True)
    shutil.copy(im_path, des)



