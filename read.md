# Hello! :)

## Navigation
- *main.py* contains the working program.
- *scrap.py* contains the functions in main.py, as well as some functions for testing.
- The folder *SEMImgTest* contains some images that can be used to test the program.

## Input and Output
This program requires three inputs: An image (taken from the SEM), the coordinates of the image (in mm), and the number of points you would like.

It will provide you with the following output: A number of .rec files (depending on how many points you asked for), and each can be loaded manually into the SEM. 

## Installations
In order for the code to properly run, the following python libraries should be installed:
- Sympy
- OpenCV
- Numpy
- Pytesseract

Plus, Tesseract OCR should be installed.

## Potential Errors
Note that there will most likely be a margin of error with the coordinates, so some manual adjustments might need to be made when viewing the specimen in the SEM.

However, the main issue is currently the contour detection from which the points are extracted. This may be due to the fact that when binary thresholding is applied to the image to prepare it for contour detection, the optimal threshold level is slightly different with each image. Images with a high amount of contrast between the background and the object in question are highly preferred.

This program is designed to work with a specific type of SEM at MIT, and as such, may not work with all SEM images. Images where the curvature of the curve is visible are preferred. 


Thank you,

*Nundhini Mascarenhas* <br>
*August 15th, 2025*