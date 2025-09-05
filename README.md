# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features :
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used :
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use :
- Clone this repository.
- Add your passport-sized photo to the images folder.
- Run the script to see your "cool" transformation!

## Applications :
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

## Program :
```
Developed by : NITHYA D
Register number : 212223240110
```
```
import cv2
import matplotlib.pyplot as plt
import numpy as np
faceImage = cv2.imread('C11791.jpg')
plt.imshow(faceImage[:,:,::-1]); plt.title("Face"); plt.show()
glassPNG = cv2.imread('sunglass.png', cv2.IMREAD_UNCHANGED)
y1, y2 = 75, 160
x1, x2 = 75, 200
roi_height = y2 - y1
roi_width = x2 - x1
glassPNG = cv2.resize(glassPNG, (roi_width, roi_height))
print("Resized Sunglasses Dimension = {}".format(glassPNG.shape))
if glassPNG.shape[2] == 4:  
    glassBGR, glassMask1 = glassPNG[:,:,:3], glassPNG[:,:,3]  
    inv_mask = cv2.bitwise_not(glassMask1)  
    glassBGR = cv2.add(
        cv2.bitwise_and(glassBGR, glassBGR, mask=glassMask1),
        cv2.bitwise_and(np.full_like(glassBGR, 255), np.full_like(glassBGR, 255), mask=inv_mask)
    )
else:  
    glassBGR = glassPNG  
    _, glassMask1 = cv2.threshold(cv2.cvtColor(glassBGR, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)  
plt.figure(figsize=[15,15])
plt.subplot(121); plt.imshow(glassBGR[:,:,::-1]); plt.title('Sunglass Color channels (White BG)')
plt.subplot(122); plt.imshow(glassMask1, cmap='gray'); plt.title('Sunglass Mask')
plt.show()
faceWithGlassesNaive = faceImage.copy()
faceWithGlassesNaive[y1:y2, x1:x2] = glassBGR
plt.imshow(faceWithGlassesNaive[...,::-1]); plt.title("Naive Overlay"); plt.show()
glassMask = cv2.merge((glassMask1, glassMask1, glassMask1))
glassMask = glassMask.astype(float) / 255.0
faceWithGlassesArithmetic = faceImage.copy()
eyeROI = faceWithGlassesArithmetic[y1:y2, x1:x2].astype(float)
maskedEye = (eyeROI * (1 - glassMask))
maskedGlass = (glassBGR.astype(float) * glassMask)
eyeRoiFinal = maskedEye + maskedGlass
faceWithGlassesArithmetic[y1:y2, x1:x2] = np.uint8(eyeRoiFinal)
plt.figure(figsize=[20,20])
plt.subplot(131); plt.imshow(maskedEye.astype(np.uint8)[...,::-1]); plt.title("Masked Eye Region")
plt.subplot(132); plt.imshow(maskedGlass.astype(np.uint8)[...,::-1]); plt.title("Masked Sunglass Region")
plt.subplot(133); plt.imshow(faceWithGlassesArithmetic[:,:,::-1]); plt.title("Augmented Eye and Sunglass")
plt.show()
plt.figure(figsize=[20,20])
plt.subplot(121); plt.imshow(faceImage[:,:,::-1]); plt.title("Original Image")
plt.subplot(122); plt.imshow(faceWithGlassesArithmetic[:,:,::-1]); plt.title("With Sunglasses")
plt.show()
```

## Output :
<img width="344" height="435" alt="image" src="https://github.com/user-attachments/assets/9feaeb4b-7f8a-4815-b535-92d257e9e0f1" />
<img width="1209" height="425" alt="image" src="https://github.com/user-attachments/assets/4db73d0b-ea23-48cd-bfa8-2d64952f7e19" />
<img width="344" height="435" alt="image" src="https://github.com/user-attachments/assets/d3b702be-95da-4804-8ee2-41b2acd12455" />
<img width="1597" height="650" alt="image" src="https://github.com/user-attachments/assets/58596352-326e-430d-a6c6-2ac2c2013f53" />
<img width="1606" height="969" alt="image" src="https://github.com/user-attachments/assets/ef582dca-e94e-4b14-ac21-6a3dda8f9e77" />
