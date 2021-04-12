import os
import cv2
import numpy as np
import SimpleITK as sitk


def Convert_To_2D(img):
    list_of_2D_images = [img[:,:,i] for i in range(1)]
    img = list_of_2D_images[0]
    return img

def Convert_To_2D_UInt8(img):
    # img = Convert_To_2D(img)
    imgArr = sitk.GetArrayFromImage(img)
    imgArrNormalized = cv2.normalize(imgArr,imgArr,0,255,cv2.NORM_MINMAX)
    imgArrUInt8 = imgArrNormalized.astype(np.uint8)
    img = sitk.GetImageFromArray(imgArrUInt8)
    return img

currentDir = os.getcwd()
movingFolderName = "Moving"
referenceFolderName = "Reference"
referenceImage = "refLung_001.dcm"
segmentationImage = "segmentation.mha"

movingFolder = os.path.join(currentDir,movingFolderName) 
referenceFolder = os.path.join(currentDir,referenceFolderName) 

movingFolderList = os.listdir(movingFolder)

movImg_array = []

refImg = sitk.ReadImage(os.path.join(currentDir,referenceFolderName,referenceImage),sitk.sitkFloat32)
refImg = Convert_To_2D(refImg)

segImg = sitk.ReadImage(os.path.join(currentDir,referenceFolderName,segmentationImage),sitk.sitkFloat32)
segImg = Convert_To_2D(segImg)

x=0

for subFolder in movingFolderList:
    if("Moving" in subFolder):
        subMoveFolder = os.path.join(movingFolder,subFolder)
        for filename in os.listdir(subMoveFolder):
            if filename.endswith(".dcm"):
                imagePath = os.path.join(subMoveFolder,filename)
                img = sitk.ReadImage(imagePath,sitk.sitkFloat32)
                movImg_array.append(Convert_To_2D(img))
                x=x+1
        #         if(x>4):
        #             break
        # if(x>4):
        #     break

# sitk.Show(movImg_array[0])
# print(refImg.GetPixelIDTypeAsString())
# print(segImg.GetPixelIDTypeAsString())
# sitk.Show(refImg)
# sitk.Show(segImg)

# Register and find the transformation
movingTransformations = []
movingTransformedMasks = []

def command_iteration(method):
    print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f} : {method.GetOptimizerPosition()}")

def findRegistration(fixed, moving):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200)
    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)

    # R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    outTx = R.Execute(fixed, moving)
    return outTx

for movingImage in movImg_array:
    movingTransformations.append(findRegistration(refImg,movingImage))


def Apply_Transform_To_Mask(ref,mask,transform):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(mask)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(transform)
    out = resampler.Execute(mask)
    return out

for transform in movingTransformations:
    movingTransformedMasks.append(Apply_Transform_To_Mask(refImg,segImg,transform))

# for i in range(len(movImg_array)):
#     movingTransformedMasks.append(Apply_Transform_To_Mask(movImg_array[i],segImg,movingTransformations[i]))


movImgUInt8_array = []
segMaskUInt8_array = []

for moving in movImg_array:
    movImgUInt8_array.append(Convert_To_2D_UInt8(moving))

for mask in movingTransformedMasks:
    segMaskUInt8_array.append(Convert_To_2D_UInt8(mask))

green = [0,255,0]

def Apply_Segmentation_Mask(ref,seg):
    original_spacing = ref.GetSpacing()
    original_size = ref.GetSize()
    min_spacing = min(original_spacing)
    new_spacing = [min_spacing, min_spacing]
    new_size = [int(round(original_size[0]*(original_spacing[0]/min_spacing))),int(round(original_size[1]*(original_spacing[1]/min_spacing)))]
    resampled_img = sitk.Resample(ref, new_size, sitk.Transform(),sitk.sitkLinear, ref.GetOrigin(),new_spacing, ref.GetDirection(), 0.0,ref.GetPixelID())
    resampled_msk = sitk.Resample(seg, new_size, sitk.Transform(),sitk.sitkNearestNeighbor, seg.GetOrigin(),new_spacing, seg.GetDirection(), 0.0,seg.GetPixelID())
    contour_overlaid_image = sitk.LabelMapContourOverlay(sitk.Cast(resampled_msk, sitk.sitkLabelUInt8),sitk.Cast(resampled_img,sitk.sitkUInt8),opacity = 1,contourThickness=[3,3],colormap=green)
    return contour_overlaid_image
                                    
maskAppliedMoving = []                                   
for i in range(len(movImgUInt8_array)):
    mAM = Apply_Segmentation_Mask(movImgUInt8_array[i],segMaskUInt8_array[i])
    maskAppliedMoving.append(mAM)

size = sitk.GetArrayFromImage(maskAppliedMoving[0]).shape
maskAppliedMovingArr = []
for imge in maskAppliedMoving:
    maskAppliedMovingArr.append(sitk.GetArrayFromImage(imge))

out = cv2.VideoWriter('projectFinalMovie.avi', cv2.VideoWriter_fourcc(*'DIVX'),15,size[:2])

for i in range(len(maskAppliedMovingArr)):
    out.write(maskAppliedMovingArr[i])
out.release()

