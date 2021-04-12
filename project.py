import os
import cv2
import pydicom as dicom
import imageio
import numpy as np
# import SimpleITK as sitk


currentDir = os.getcwd()
movingFolderName = "Moving"

movingFolder = os.path.join(currentDir,movingFolderName) 
movingFolderList = os.listdir(movingFolder)


img_array = []
dicom_img = []
size = 0
for subFolder in movingFolderList:
    if("Moving" in subFolder):
        subMoveFolder = os.path.join(movingFolder,subFolder)
        for filename in os.listdir(subMoveFolder):
            if filename.endswith(".dcm"):
                imagePath = os.path.join(subMoveFolder,filename)
                img = dicom.dcmread(imagePath)
                dicom_img.append(img)

                
                img_uint8 = cv2.normalize(img.pixel_array,img.pixel_array,0,255,cv2.NORM_MINMAX)
                img_uint8 = (img.pixel_array).astype(np.uint8)
                img_uint8 = img_uint8

                size = img.pixel_array.shape

                jpgImgName = os.path.join(subMoveFolder,filename+'.jpg')
                imageio.imwrite(jpgImgName, img_uint8)

                jpgimg = cv2.imread(jpgImgName)
                img_array.append(jpgimg)

out = cv2.VideoWriter('projectInitMovie.avi', cv2.VideoWriter_fourcc(*'DIVX'),15,size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

