import SimpleITK as sitk
import sys
import os
  
refImg = "D:\\OneDrive - Wonka\\Subjects\\CAI\\CAI-FinalProject\\Abdominal_Data\\Reference\\refLung_001.dcm"
movImg = "D:\\OneDrive - Wonka\\Subjects\\CAI\\CAI-FinalProject\\Abdominal_Data\\Moving\\Moving_001\\Moving_001.dcm"

def command_iteration(method):
    print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f} : {method.GetOptimizerPosition()}")

fixed = sitk.ReadImage(refImg, sitk.sitkFloat32)
list_of_2DFixed_images = [fixed[:,:,i] for i in range(1)]

moving = sitk.ReadImage(movImg, sitk.sitkFloat32)
list_of_2DMoving_images = [moving[:,:,i] for i in range(1)]

R = sitk.ImageRegistrationMethod()
R.SetMetricAsMeanSquares()
R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200)
R.SetInitialTransform(sitk.TranslationTransform(list_of_2DFixed_images[0].GetDimension()))
R.SetInterpolator(sitk.sitkLinear)

R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

outTx = R.Execute(list_of_2DFixed_images[0], list_of_2DMoving_images[0])

print("-------")
print(outTx)
print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
print(f" Iteration: {R.GetOptimizerIteration()}")
print(f" Metric value: {R.GetMetricValue()}")

sitk.WriteTransform(outTx, 'transofrmInit.hdf5')

if ("SITK_NOSHOW" not in os.environ):
     resampler = sitk.ResampleImageFilter()
     resampler.SetReferenceImage(list_of_2DFixed_images[0])
     resampler.SetInterpolator(sitk.sitkLinear)
     resampler.SetDefaultPixelValue(100)
     resampler.SetTransform(outTx)
  
     out = resampler.Execute(list_of_2DMoving_images[0])
     simg1 = sitk.Cast(sitk.RescaleIntensity(list_of_2DFixed_images[0]), sitk.sitkUInt8)
     simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
     cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
     sitk.Show(cimg, "ImageRegistration1 Composition")