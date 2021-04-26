# Move the registerMoving.py to the Abdominal_Data folder and then run it.
# Program runtime - almost 15 seconds

import os #python 3 default package
import sys #python 3 default package
import cv2 #pip install opencv-python
import numpy as np #pip install numpy
import SimpleITK as sitk #pip install simpleitk

import time #python 3 default package
start_time = time.time()

def Convert_To_2D(img): #Converts 3D images to 2D
    list_of_2D_images = [img[:,:,i] for i in range(1)]
    img = list_of_2D_images[0]
    return img

def Convert_To_2D_UInt8(img):#Converts any image to UInt8 pixel type
    imgArr = sitk.GetArrayFromImage(img)
    imgArrNormalized = cv2.normalize(imgArr,imgArr,0,255,cv2.NORM_MINMAX) #normalize the value so that the conversion is lossless
    imgArrUInt8 = imgArrNormalized.astype(np.uint8)
    img = sitk.GetImageFromArray(imgArrUInt8)
    return img

if len(sys.argv) != 5:
    print('\nUsage: ' + sys.argv[0] + ' <PathToMovingFolder> <PathToReferenceFolder> <referenceImageName> <segmentationImageName>')
    print('\nEx: python ' + sys.argv[0] + r" C:\Users\Abdominal_Data\Moving C:\Users\Abdominal_Data\Reference refLung_001.dcm segmentation.mha")
    sys.exit(1)

#Putting the code inside the Abdominal_Data folder and running should work
movingFolder = sys.argv[1] #ex - C:\Users\Abdominal_Data\Moving
referenceFolder = sys.argv[2] #ex - C:\Users\Abdominal_Data\Reference
referenceImage = sys.argv[3] #ex - refLung_001.dcm
segmentationImage = sys.argv[4] #ex - segmentation.mha


movingFolderList = os.listdir(movingFolder)


#loding the reference image from /Reference
refImg = sitk.ReadImage(os.path.join(referenceFolder,referenceImage),sitk.sitkFloat32)
refImg = Convert_To_2D(refImg)

#loding the segmentation mask image from /Reference
segImg = sitk.ReadImage(os.path.join(referenceFolder,segmentationImage),sitk.sitkFloat32)
segImg = Convert_To_2D(segImg)



movImg_array = [] #array to store all the moving image

#loop through the contents of the /Moving folder and read all the moving images
for subFolder in movingFolderList:
    if("Moving" in subFolder):
        subMoveFolder = os.path.join(movingFolder,subFolder)
        for filename in os.listdir(subMoveFolder):
            if filename.endswith(".dcm"):
                imagePath = os.path.join(subMoveFolder,filename)
                img = sitk.ReadImage(imagePath,sitk.sitkFloat32)
                movImg_array.append(Convert_To_2D(img))


###Now that we have loaded all the input images lets start the registration work

# <STARTING IMAGE REGISTRATION>

# Register and find the transformation
movingTransformations = [] #array to store the calculated transformations
movingTransformedMasks = [] #array to store the masks after applying the calculated transformation on them
transformationStoppingMetrics = [] #array to store the final stopping metric value of each tranformation

#Registration method
def findRegistration(fixed, moving,fixed_image_mask, lastTransform):
    R = sitk.ImageRegistrationMethod() #create the registration method object
    R.SetMetricAsMeanSquares() #set the comparission metric as mean sqares
    R.SetMetricFixedMask(fixed_image_mask) #inform the regitration object about the area to be registered
    # R.SetMetricSamplingPercentage(0.7, sitk.sitkWallClock)
    # R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetOptimizerAsRegularStepGradientDescent(1, .001, 1000) #set the model parameters
    R.SetInitialTransform(lastTransform) #initialie the transform with the previous transform
    R.SetInterpolator(sitk.sitkLinear) #using the linear interpolator
    final_transform = R.Execute(fixed, moving)    #execute the registration method on our fixed and moving image
    transformationStoppingMetrics.append(R.GetOptimizerIteration())
    return final_transform

#loop through the moving image array and find its transform with refernce to the reference image and fixed segmentation mask image
lastTransform = sitk.TranslationTransform(refImg.GetDimension())
for movingImage in movImg_array:
    lastTransform = findRegistration(refImg,movingImage,segImg, lastTransform)
    movingTransformations.append(lastTransform)

meanTransformationStoppingMetric = np.mean(transformationStoppingMetrics)
print('Optimizer\'s mean stopping iteration : {0}'.format(meanTransformationStoppingMetric))

# <END OF IMAGE REGISTRATION>


# <APPLYING THE TRANSFORM TO THE MASK>
def Apply_Transform_To_Mask(ref,mask,transform): #applies the transform obtained for each moving image to the segmentation mask
    out = sitk.Resample(mask, ref, transform, sitk.sitkLinear, 0.0, mask.GetPixelID())
    return out

# loop through the transforms and apply each transform to the fixed segementation mask and then store them in movingTransformedMasks
for transform in movingTransformations:
    movingTransformedMasks.append(Apply_Transform_To_Mask(refImg,segImg,transform.GetInverse()))
# <FINISHED APPLYING THE TRANSFORM TO THE MASK>


# <APPLYING THE TRANSFORMED MASK TO THE MOVING IMAGE>
movImgUInt8_array = []
segMaskUInt8_array = []

#convert the moving images to uint8
for moving in movImg_array:
    movImgUInt8_array.append(Convert_To_2D_UInt8(moving))

#convert the transformed mask of each moving images to uint8
for mask in movingTransformedMasks:
    segMaskUInt8_array.append(Convert_To_2D_UInt8(mask))

green = [0,255,0]

def Apply_Segmentation_Mask(ref,seg,transform): #applies a segementation mask on an image
    contour_overlaid_image = sitk.LabelMapContourOverlay(sitk.Cast(seg, sitk.sitkLabelUInt8),ref,opacity = 1,contourThickness=[1,1],colormap=green)
    return contour_overlaid_image

# loop through the moving images and then apply their corresponding mask on it for segmentation                             
maskAppliedMoving = []                                   
for i in range(len(movImgUInt8_array)):
    mAM = Apply_Segmentation_Mask(movImgUInt8_array[i],segMaskUInt8_array[i],movingTransformations[i])
    maskAppliedMoving.append(mAM)
# <FINISHED APPLYING THE TRANSFORMED MASK TO THE MOVING IMAGE>

# <CREATING THE FINAL VIDEO>
#extract the array from the segmented moving image to write them to the video
size = sitk.GetArrayFromImage(maskAppliedMoving[0]).shape
maskAppliedMovingArr = []
for imge in maskAppliedMoving:
    maskAppliedMovingArr.append(sitk.GetArrayFromImage(imge))

# initialize a new video
out = cv2.VideoWriter('finalMovie.avi', cv2.VideoWriter_fourcc(*'DIVX'),15,size[:2])

# write all the segmented moving images to the video and release it.
for i in range(len(maskAppliedMovingArr)):
    out.write(maskAppliedMovingArr[i])
out.release()
# <FINISHED CREATING THE FINAL VIDEO>

print("--- Tracking video created in %s seconds ---" % (time.time() - start_time))
