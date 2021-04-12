import SimpleITK as sitk
import numpy as np

import sys
import cv2

import os
import pydicom as dicom

  
# refImg = "D:\\OneDrive - Wonka\\Subjects\\CAI\\CAI-FinalProject\\Abdominal_Data\\Reference\\refLung_001.dcm"
refImg = "D:\\OneDrive - Wonka\\Subjects\\CAI\\CAI-FinalProject\\Abdominal_Data\\Moving\\Moving_001\\Moving_001.dcm"

segImg = "D:\\OneDrive - Wonka\\Subjects\\CAI\\CAI-FinalProject\\Abdominal_Data\\Reference\\segmentation.mha"

ref = sitk.ReadImage(refImg)
list_of_2DFRef_images = [ref[:,:,i] for i in range(1)]
ref = list_of_2DFRef_images[0]

refArr = sitk.GetArrayFromImage(ref)
refArrNormalized = cv2.normalize(refArr,refArr,0,255,cv2.NORM_MINMAX)
refArrUInt8 = refArrNormalized.astype(np.uint8)
ref = sitk.GetImageFromArray(refArrUInt8)
# sitk.Show(ref)


seg = sitk.ReadImage(segImg)
list_of_2DSeg_images = [seg[:,:,i] for i in range(1)]
seg = list_of_2DSeg_images[0]

segArr = sitk.GetArrayFromImage(seg)
segArrNormalized = cv2.normalize(segArr,segArr,0,255,cv2.NORM_MINMAX)
segArrUInt8 = segArrNormalized.astype(np.uint8)
seg = sitk.GetImageFromArray(segArrUInt8)
# sitk.Show(seg)




red = [255,0,0]
green = [0,255,0]
blue = [0,0,255]


original_spacing = ref.GetSpacing()
original_size = ref.GetSize()
min_spacing = min(original_spacing)
new_spacing = [min_spacing, min_spacing]
new_size = [int(round(original_size[0]*(original_spacing[0]/min_spacing))),
            int(round(original_size[1]*(original_spacing[1]/min_spacing)))]
resampled_img = sitk.Resample(ref, new_size, sitk.Transform(),
                                sitk.sitkLinear, ref.GetOrigin(),
                                new_spacing, ref.GetDirection(), 0.0,
                                ref.GetPixelID())

resampled_msk = sitk.Resample(seg, new_size, sitk.Transform(),
                                sitk.sitkNearestNeighbor, seg.GetOrigin(),
                                new_spacing, seg.GetDirection(), 0.0,
                                seg.GetPixelID())



contour_overlaid_image = sitk.LabelMapContourOverlay(sitk.Cast(resampled_msk, sitk.sitkLabelUInt8),
                                       sitk.Cast(resampled_img,sitk.sitkUInt8),
                                       opacity = 1,
                                       contourThickness=[3,3],colormap=green)


sitk.Show(contour_overlaid_image)
# print(contour_image)
