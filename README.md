# Medical segment tracking

A computer aided intervention tool to track a segment of a scan across multiple moving images.

When it comes to automated medical intervention, making a computer identify and fix a particular probem is not far to be common.

However, the biological organs, have a lot of movement and is to be handled very carefully.

To do this, the intervening computer should have the ability to track the area of interest it is working on.

This project bootstraps a tool to achieve that.

Input required :
- A reference image of the organ. (/Reference/referenceLungScan.dcm)
- A segmentation mask highlighting the area to be tracked w.r.t the reference image. (/Reference/segmentationMask.mha)
- The moving images (/Moving/...)

Output :
- A video of all the moving image arranged in sequence with the area of interest tracked.

The tool was developed with reference to tracking a defective area of human lungs.
The sample data are provided along with. 

Requirements:
- [pip install simpleitk](https://pypi.org/project/SimpleITK/)
- [pip install opencv-python](https://pypi.org/project/opencv-python/)

How to run:
- The tool require four inputs -> <PathToMovingFolder> <PathToReferenceFolder> <referenceImageName> <segmentationImageName>

Example:
- python trackSegment.py 'C:\Users\Desktop\Moving' 'C:\Users\Desktop\Reference' referenceLungScan.dcm segmentationMask.mha