#notes, all func have the boolean kwarg "is_test" automatically set to False
#if set to True, it will print some additional info to the console for debugging
#this file is meant for testing the functions

import sympy as sp
import math
import cv2 as cv
import numpy as np
#from PIL import Image
import pytesseract
from fractions import Fraction

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\moona\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

def createCircle(constantsOfCir, numOfPoints, is_test=False): #takes in list of the constants of a circle, an int number of points, prints a list of coords
    h = constantsOfCir[0]
    k = constantsOfCir[1]
    r = constantsOfCir[2]
    rSquared = r * r
    if is_test:
        print(f"Your equation is: (x-{h})^2 + (y-{k})^2 = {rSquared}")
    points = []
    incremental_angle = (2*sp.pi) / numOfPoints
    angle = incremental_angle
    if is_test:
        print(f"The angle we're incrementing by in radians is {incremental_angle}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        angles_used = []
    try:
        for currentPoint in range(numOfPoints):
            x = (r * sp.cos(angle)) + h
            y = (r * sp.sin(angle)) + k #sin of pi is 0, cos is -1
            currentPairOfCoords = [x, y]
            points.append(currentPairOfCoords)
            if is_test:
                angles_used.append(angle)
            angle += incremental_angle
    except TypeError:
        print("Number of points must be an integer")
    
    else:
        #print(points)
        if is_test:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Here are the angles that were used: ")
            print(angles_used)

        if is_test:
            approxPoints = []
            for point in points:
                x = point[0].evalf()
                y = point[1].evalf()
                approxCoords =  [x, y]
                approxPoints.append(approxCoords)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Here are the approximate coords for testing: ")
            print(approxPoints)
        return points

def solveSim(is_test=False): #used MCC Py Tutorial vid on yt
    x = sp.Symbol("x")
    eqn = sp.Eq(x+9,8)
    print(eqn)
    print(sp.solve(eqn))
    
def solveCircle(points, is_test=False): #takes list containing three ordered pairs, outputs list containing the constants of a circl
    #points should be in form [x1, y1, x2, y2, x3, y3]
    x1 = points[0]
    y1 = points[1]
    x2 = points[2]
    y2 = points[3]
    x3 = points[4]
    y3 = points[5]

    try:
        x12 = x1 - x2; 
        x13 = x1 - x3; 

        y12 = y1 - y2; 
        y13 = y1 - y3; 
    except TypeError:
        print("All coordinates must be numbers")
    except:
        print("An unexpected error happened while finding the constants of the circle (See solveCircle())")
    else:
        y31 = y3 - y1; 
        y21 = y2 - y1; 

        x31 = x3 - x1; 
        x21 = x2 - x1; 

        sx13 = pow(x1, 2) - pow(x3, 2); 
        # y1^2 - y3^2 
        sy13 = pow(y1, 2) - pow(y3, 2); 
        
        sx21 = pow(x2, 2) - pow(x1, 2); 
        sy21 = pow(y2, 2) - pow(y1, 2); 

        f = (((sx13) * (x12) + (sy13) * 
            (x12) + (sx21) * (x13) + 
            (sy21) * (x13)) / (2 * 
            ((y31) * (x12) - (y21) * (x13))))
                
        g = (((sx13) * (y12) + (sy13) * (y12) + 
            (sx21) * (y13) + (sy21) * (y13)) / 
            (2 * ((x31) * (y12) - (x21) * (y13))))

        c = (-pow(x1, 2) - pow(y1, 2) - 
            2 * g * x1 - 2 * f * y1)

        h = -g
        k = -f 
        sqr_of_r = h * h + k * k - c

        r = sp.sqrt(sqr_of_r)
        constantsOfCir = [h, k, r]

        if is_test:
            print(f"Center of circle is ({h}, {k})")
            print(f"Radius is {r}")
            print(f"Constants are {constantsOfCir}")
        return constantsOfCir

def stitch(is_test=False):
    if is_test:
        print("Wassup") #test message to check if the program works
    current_dir = Path("ImgTest") 
    imgs = []
    for item in current_dir.iterdir():
        imgs.append(cv.imread(item)) #adds every

    stitcher = cv.Stitcher.create()
    (status, stitched) = stitcher.stitch(imgs) #stitches together images

    if status == cv.Stitcher_OK:
        cv.imwrite("TestResults/newTest.jpg", stitched) #downloads stitched together image
    else:
        print(f"Bruh: {status}") #prints error message

def findCurves(img, is_test=False): #takes in an image and it's coordinates, outputs 3 represenative points of a circle
    
    if is_test:
        cv.imshow('Here is the Image', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    greyscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if is_test:
        cv.imshow('Here is the greyscale Image', greyscale_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    threshold = 90
    ret, binary_img = cv.threshold(greyscale_img, threshold, 255, cv.THRESH_BINARY)

    if is_test:
        cv.imshow(f"Here is the Binary Image with theshold {threshold}", binary_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    contours, hierarchy = cv.findContours(image=binary_img, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    
    if is_test:
        img_copy = img.copy()
        cv.drawContours(image=img_copy, contours=contours, contourIdx=-1, color=(0,255,0), thickness=2, lineType=cv.LINE_AA)
        cv.imshow("Here are the contours", img_copy)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    img_copy = img.copy()
    potentialContours = []
    
    if is_test:
        print(f"{len(contours)} contours were found")
        sizes_of_contours = []
    
    cutoffContourSize = 100000
    for i in range(len(contours)):
        if cv.contourArea(contours[i]) > cutoffContourSize:
            potentialContours.append(i)
        
        if is_test:
            cv.drawContours(image = img_copy, contours=contours, contourIdx=i, color=(0,255,0), thickness=2, lineType=cv.LINE_AA)
            cv.imshow(f"Contour #{i}", img_copy)
            cv.waitKey(0)
            cv.destroyAllWindows()
            img_copy = img.copy()
            sizes_of_contours.append(cv.contourArea(contours[i]))
    
    if is_test:
        print(potentialContours)
        print(sizes_of_contours)
    
    #BREAK

    largest_contour = max(potentialContours)
    
    if is_test:
        print(f"This is longest contours{contours[largest_contour]}")
        print(f"It is {len(contours[largest_contour])} points long")
    
    working_contour = contours[largest_contour]
    rim = []
    for i in range(len(working_contour)):
        previous_x = working_contour[i-1][0][0] if i > 0 else working_contour[-1][0][0]
        next_x = working_contour[i+1][0][0] if i < len(working_contour) - 1 else working_contour[0][0][0]

        current_x = working_contour[i][0][0]

        if not (current_x == previous_x == next_x):
            rim.append(working_contour[i][0])
    
    for point in rim:
        cv.circle(img_copy, point, radius=2, color=(0, 255, 0), thickness=-1)
    
    if is_test:
        cv.imshow("Potential Rims", img_copy)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    contour_0 = []
    contour_1 = []
    current_contour = 0
    total = 0
    cutoff = 15 #fix name
    for i in range(len(rim)):
        current_y = rim[i][1]
        next_y = rim[i+1][1] if i < (len(rim) - 1) else rim[0][1]
        distance_between_points = abs(next_y - current_y)
        if distance_between_points > cutoff:
            current_contour ^= 1
        if current_contour == 1:
            contour_1.append(rim[i])
        else:
            contour_0.append(rim[i])
    
    if is_test:
        img_copy = img.copy()
        for point in contour_0:
            cv.circle(img_copy, point, radius=2, color=(0, 255, 0), thickness=-1)
        cv.imshow("Contour 0", img_copy)
        cv.waitKey(0)
        cv.destroyAllWindows()

        for point in contour_1:
            cv.circle(img_copy, point, radius=2, color=(0, 255, 0), thickness=-1)
        cv.imshow("Contour 1", img_copy)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    working_contour = contour_1

    trimPercent = 20
    outlier_threshold = len(working_contour) // trimPercent

    trimmed_contour = []
    for i in range(len(working_contour)):
        if i < outlier_threshold:
            pass
        elif i > (len(working_contour) - outlier_threshold):
            pass
        else:
            trimmed_contour.append(working_contour[i])

    if is_test:
        img_copy = img.copy()
        for point in trimmed_contour:
            cv.circle(img_copy, point, radius=2, color=(0, 255, 0), thickness=-1)
        cv.imshow("The trimmed contour", img_copy)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    fourthOfContour = len(trimmed_contour) // 4
    
    sampleCoords = []
    sampleCoords.append(trimmed_contour[fourthOfContour][0])
    sampleCoords.append(trimmed_contour[fourthOfContour][1])
    sampleCoords.append(trimmed_contour[fourthOfContour*2][0])
    sampleCoords.append(trimmed_contour[fourthOfContour*2][1])
    sampleCoords.append(trimmed_contour[fourthOfContour*3][0])
    sampleCoords.append(trimmed_contour[fourthOfContour*3][1])

    convertedSampleCoords = []
    for Coord in sampleCoords:
        convertedSampleCoords.append(int(Coord))
    
    return convertedSampleCoords

def readImage(img, is_test=False):
    imgCompat = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    print(pytesseract.image_to_string(imgCompat))

def trimContourAndReturnPoints(img, working_contour, is_test=False): 

    trimPercent = 20
    outlier_threshold = len(working_contour) // trimPercent

    trimmed_contour = []
    for i in range(len(working_contour)):
        if i < outlier_threshold:
            pass
        elif i > (len(working_contour) - outlier_threshold):
            pass
        else:
            trimmed_contour.append(working_contour[i])

    if is_test:
        img_copy = img.copy()
        for point in trimmed_contour:
            cv.circle(img_copy, point, radius=2, color=(0, 255, 0), thickness=-1)
        cv.imshow("The trimmed contour", img_copy)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    fourthOfContour = len(trimmed_contour) // 4
    
    sampleCoords = []
    sampleCoords.append(trimmed_contour[fourthOfContour][0])
    sampleCoords.append(trimmed_contour[fourthOfContour][1])
    sampleCoords.append(trimmed_contour[fourthOfContour*2][0])
    sampleCoords.append(trimmed_contour[fourthOfContour*2][1])
    sampleCoords.append(trimmed_contour[fourthOfContour*3][0])
    sampleCoords.append(trimmed_contour[fourthOfContour*3][1])

    convertedSampleCoords = []
    for Coord in sampleCoords:
        convertedSampleCoords.append(int(Coord))
    
    return convertedSampleCoords
def findRoughContour(img, is_test=False):
    if is_test:
        cv.imshow('Here is the starting Image', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    if len(img.shape) == 3:
        greyscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else: 
        greyscale_img = img

    if is_test:
        cv.imshow('Here is the greyscale Image', greyscale_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    height, width = greyscale_img.shape

    threshold = 90
    ret, binary_img = cv.threshold(greyscale_img, threshold, 255, cv.THRESH_BINARY)

    if is_test:
        cv.imshow(f"Here is the Binary Image with theshold {threshold}", binary_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    contours, hierarchy = cv.findContours(image=binary_img, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    
    if is_test:
        img_copy = img.copy()
        cv.drawContours(image=img_copy, contours=contours, contourIdx=-1, color=(0,255,0), thickness=2, lineType=cv.LINE_AA)
        cv.imshow("Here are the contours", img_copy)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    img_copy = img.copy()
    potentialContours = []
    
    if is_test:
        print(f"{len(contours)} contours were found")
        sizes_of_contours = []
    
    cutoffContourSize = 100000
    for i in range(len(contours)):
        if cv.contourArea(contours[i]) > cutoffContourSize:
            potentialContours.append(i)
        
        # if is_test:
        #     cv.drawContours(image = img_copy, contours=contours, contourIdx=i, color=(0,255,0), thickness=2, lineType=cv.LINE_AA)
        #     cv.imshow(f"Potential Contour whose number is #{i}", img_copy)
        #     cv.waitKey(0)
        #     cv.destroyAllWindows()
        #     img_copy = img.copy()
        #     sizes_of_contours.append(cv.contourArea(contours[i]))
    
    if is_test:
        print(f"Here are all the potential Contours: {potentialContours}")
        print(f"These are their lengths: {sizes_of_contours}")

    largest_contour = max(potentialContours)
    
    if is_test:
        print(f"This is longest contour: {contours[largest_contour]}")
        print(f"It is {len(contours[largest_contour])} points long")
    
    working_contour = contours[largest_contour]
    return working_contour
def findUseableContours(img, working_contour, is_test=False):
    rim = []
    for i in range(len(working_contour)):
        previous_x = working_contour[i-1][0][0] if i > 0 else working_contour[-1][0][0]
        previous_y = working_contour[i-1][0][1] if i > 0 else working_contour[-1][0][1]
        next_x = working_contour[i+1][0][0] if i < len(working_contour) - 1 else working_contour[0][0][0]
        next_y = working_contour[i+1][0][1] if i < len(working_contour) - 1 else working_contour[0][0][1]

        current_x = working_contour[i][0][0]
        current_y = working_contour[i][0][1]

        if not (current_x == previous_x == next_x) and not (current_y == previous_y == next_y):
            rim.append(working_contour[i][0])
       
    if is_test:
        img_copy = img.copy()
        for point in rim:
            cv.circle(img_copy, point, radius=2, color=(0, 255, 0), thickness=-1)
        cv.imshow("Potential Rims with Edges of Screen Removed", img_copy)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    contour_0 = []
    contour_1 = []
    current_contour = 0
    total = 0
    cutoff = 15 #fix name
    for i in range(len(rim)):
        current_y = rim[i][1]
        next_y = rim[i+1][1] if i < (len(rim) - 1) else rim[0][1]
        distance_between_points = abs(next_y - current_y)
        if distance_between_points > cutoff:
            current_contour ^= 1
        if current_contour == 1:
            contour_1.append(rim[i])
        else:
            contour_0.append(rim[i])
    
    if is_test:
        img_copy = img.copy()
        for point in contour_0:
            cv.circle(img_copy, point, radius=2, color=(0, 255, 0), thickness=-1)
        cv.imshow("Contour 0", img_copy)
        cv.waitKey(0)
        cv.destroyAllWindows()

        img_copy = img.copy()
        for point in contour_1:
            cv.circle(img_copy, point, radius=2, color=(0, 255, 0), thickness=-1)
        cv.imshow("Contour 1", img_copy)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    cvcontour_0 = np.array(contour_0, dtype=np.float32)
    cvcontour_1 = np.array(contour_1, dtype=np.float32)
    (cir_x0, cir_y0), radius0 = cv.minEnclosingCircle(cvcontour_0)
    (cir_x1, cir_y1), radius1 = cv.minEnclosingCircle(cvcontour_1)

    origin0 = np.array([cir_x0, cir_y0])
    origin1 = np.array([cir_x1, cir_y1])

    totalDevi0 = 0
    totalDevi1 = 0
    avgDevi0 = 0
    avgDevi1 = 0

    for point in contour_0:
        deviFromRad = np.linalg.norm(point - origin0)
        totalDevi0 += abs(1 - deviFromRad)
    for point in contour_1:
        deviFromRad = np.linalg.norm(point - origin1)
        totalDevi1 += abs(1 - deviFromRad)
    avgDevi0 = totalDevi0 / len(contour_0)
    avgDevi1 = totalDevi1 / len(contour_1)

    if avgDevi0 < avgDevi1: 
        if is_test:
            print("contour_0 used")
        return contour_0
    else:
        if is_test:
            print("contour_1 used")
        return contour_1  
def findScale(img, height, width, is_test=False):
    threshold = 50 #amount of non-black pixels allowed in a row, any more and the program will not consider it a black line
    first_line_passed = False
    top_of_key = int(height*.9)
    scale_bar_location = int(width - (width*0.40))
    # bottom_of_key = 0

    # for y in range(height-1, -1, -1):
    #     row = img[y,:]
    #     if cv.countNonZero(row) < threshold:
    #         if not first_line_passed:
    #             first_line_passed = True
    #             bottom_of_key = y
    #             if is_test:
    #                 print(y)
    #         elif first_line_passed:
    #             top_of_key = y
    #             if is_test:
    #                 print(y)
    #             break
    
    key_img = img[top_of_key:height, scale_bar_location:width]
    #[top_of_key:bottom_of_key, 0:int(width*0.25)]
    if len(key_img.shape) == 3:
        greyscale_keyimg = cv.cvtColor(key_img, cv.COLOR_BGR2GRAY)
    else: 
        greyscale_keyimg = key_img

    if is_test:
        cv.imshow("This the the cropped img with only the key", key_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    threshold = 250
    ret, binary_key = cv.threshold(greyscale_keyimg, threshold, 255, cv.THRESH_BINARY)

    if is_test:
        cv.imshow(f"Here is the Binary Image with theshold {threshold}", binary_key)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    contours, hierarchy = cv.findContours(image=binary_key, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_NONE)
    if len(img.shape) == 2:
        key_img = cv.cvtColor(binary_key, cv.COLOR_GRAY2BGR)

    if is_test:
        img_copy = key_img.copy()
        cv.drawContours(image=img_copy, contours=contours, contourIdx=-1, color=(0,255,0), thickness=2, lineType=cv.LINE_AA)
        cv.imshow("Here are the contours", img_copy)
        cv.waitKey(0)
        cv.destroyAllWindows()

    if is_test:
        if len(contours) <= 20:
            for i in range(len(contours)):
                img_copy = key_img.copy()
                cv.drawContours(image=img_copy, contours=contours, contourIdx=i, color=(0,255,0), thickness=2, lineType=cv.LINE_AA)
                cv.imshow(f"This is contour #{i}", img_copy)
                cv.waitKey(0)
                cv.destroyAllWindows()
        else:
            print("Too many contours to display")
    
    potential_contours = list(contours)
    # for contour in contours:
    #     x, y, w, h = cv.boundingRect(contour)
    #     aspect_ratio = w / h
    #     if aspect_ratio > 2.5:
    #         potential_contours.append(contour)
    
    # if is_test:
    #     if len(potential_contours) <= 20:
    #         print(f"{len(potential_contours)} found")
    #         for i in range(len(potential_contours)):
    #             img_copy = key_img.copy()
    #             cv.drawContours(image=img_copy, contours=potential_contours, contourIdx=i, color=(0,255,0), thickness=2, lineType=cv.LINE_AA)
    #             cv.imshow(f"This is potential contour #{i}", img_copy)
    #             cv.waitKey(0)
    #             cv.destroyAllWindows()
    #     else:
    #         print("Too many contours to display")
    
    # scaleBar = max(range(len(potential_contours)), key=lambda i: cv.contourArea(potential_contours[i]))
    # del potential_contours[indexOfBorder]
    scaleBar = max(potential_contours, key=cv.contourArea)

    if is_test:
        img_copy = key_img.copy()
        cv.drawContours(image=img_copy, contours=[scaleBar], contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
        cv.imshow(f"This is the scale bar", img_copy)
        cv.waitKey(0)
        cv.destroyAllWindows()

    horizontal_bar = []
    for point in range(len(scaleBar)):
        current_x, current_y = scaleBar[point][0]
        next_x, next_y = scaleBar[point+1][0] if point + 1 < len(scaleBar) else scaleBar[0][0]
        prev_x, prev_y = scaleBar[point-1][0] if point - 1 > -1 else scaleBar[len(scaleBar)-1][0]
        if prev_y == current_y == next_y and current_x != next_x and current_x != prev_x:
            horizontal_bar.append(scaleBar[point])

    if is_test:
        img_copy = key_img.copy()
        for point in horizontal_bar:
            x, y = point[0]
            cv.circle(img_copy, (x,y), radius=3, color=(0,255,0), thickness=-1)
        cv.imshow(f"This is the horizontal scale bar", img_copy)
        cv.waitKey(0)
        cv.destroyAllWindows()

    leftMostValue = width #just to make sure all values will be less than the width
    rightMostValue = 0
    for point in range(len(horizontal_bar)):
        x_value = horizontal_bar[point][0][0]
        leftMostValue = min(leftMostValue, x_value)
        rightMostValue = max(rightMostValue, x_value)
    lengthOfScaleBar = abs(rightMostValue-leftMostValue)
    if is_test:
        print(f"Pixel length of scale bar is {lengthOfScaleBar}")

    tessCompatKeyImg = cv.cvtColor(key_img, cv.COLOR_BGR2RGB)
    tessCompatKeyImg = tessCompatKeyImg[:, :rightMostValue+30]
    if is_test:
        cv.imshow("Text Image", tessCompatKeyImg)
        cv.waitKey(0)
        cv.destroyAllWindows()

    textFound = pytesseract.image_to_string(tessCompatKeyImg)
    
    piecesOfKeyText = []
    for char in textFound:
        if char == "\n":
            break
        else:
            if char =="u":
                piecesOfKeyText.append("μ")
            else:
                piecesOfKeyText.append(char)
    
    keyText = "".join(piecesOfKeyText)
    if is_test:
        print(f"Text is {keyText}")

    prevalue = []
    preunit = []
    for char in keyText:
        if char.isdigit():
            prevalue.append(char)
        elif char == " ":
            pass
        else:
            preunit.append(char)

    value = int("".join(prevalue))
    unit = "".join(preunit)
    if unit == "μm":
        is_in_microns = True
    if unit == "mm":
        is_in_microns = False
    
    scale = Fraction(value, lengthOfScaleBar) #typecast as float when necessary
    scale = scale / 1000 if is_in_microns else scale
    if is_test:
        print(f"There are {float(scale)}mm per pixel")
    

    return scale

def findRepPoints(img, is_test=False): 
    height, width = img.shape[:2]
    pixel_center_x = width // 2
    pixel_center_y = height // 2
    scale = findScale(img, height, width, is_test) #scale is in millimeters per pixel

    potentialContours = findRoughContour(img, is_test)
    working_contour = findUseableContours(img, potentialContours, is_test)
    return (trimContourAndReturnPoints(img, working_contour, is_test), scale)
    #add function that converts pixel points into real points

def convertPixelToReal(points, scale, is_test=False):
    convertedPoints = []
    for point in points:
        convertedPoint = point * float(scale)
        convertedPoints.append(convertedPoint)
    return convertedPoints

def findDistanceFromStart(points, center_points, is_test=False):
    x_center, y_center = center_points
    x_point, y_point = points
    x_mov = x_point - x_center
    y_mov = y_point - y_center
    return [x_mov, y_mov]

def createRecipeFile(mov, current_itr, targetDir="RecFiles", is_test=False):
    x_mov, y_mov = mov
    file = open(f"{targetDir}/gen{current_itr}.rec", "w")
    text = f'''[VACUUM]
        MODE=LV
        LVMODE=AUTO
        PRESS=30
        [GUN]
        ACCV=15000
        [GUNALIN]
        TILT-X=132
        TILT-Y=136
        SHIFT-X=119
        SHIFT-Y=128
        [LENS]
        SPOTSIZE=45
        CL=1201
        OLC=2259
        OLF=2182
        DFCMODE=OFF
        DFC=0
        AFC=2048
        [SCAN]
        TILTCORRECTMODE=OFF
        TILTCORRECTANGLE=0
        SRTMODE=OFF
        SRTANGLE=-79
        WDLINK=ON
        MAG=450
        FINESHIFT-X=2048
        FINESHIFT-Y=2048
        [STIGMA]
        STIG-X=2048
        STIG-Y=2180
        [SED]
        SUPPRESS=2048
        CONTRAST=1200
        BRIGHTNESS=1953
        [SIGNAL]
        SIGNAL=LVSE
        COLLECTORVOLTAGE=NORMAL
        [STAGE]
        X={x_mov}
        Y={y_mov}
        T=0.003
        Z=80.000
        R=0.000
        [DATE]
        DATE=25.08.05'''
    file.write(text)


#program can can send x y coords
#magnification
#need to covert pixel coords into real coords using real coords and magnification 

#img = cv.imread("SEMImgTest/4.tif", cv.IMREAD_GRAYSCALE)
def main():
    img = cv.imread("SEMImgTest/testReal.bmp")
    coords_of_center = (30, 20)
    numberOfPointsRequested = 6
    rep_points, scale = findRepPoints(img, is_test=True)
    constants = solveCircle(rep_points)
    pixel_points_of_circle = createCircle(constants, numberOfPointsRequested) #stored in lists of 2 all contained within a larger list
    real_points = []
    for coordPair in pixel_points_of_circle:
        real_coord = convertPixelToReal(coordPair, scale)
        real_points.append(real_coord)
    distanceToMove = []
    for point in real_points:
        distanceToMove.append(findDistanceFromStart(point, coords_of_center))

    for i in range(len(distanceToMove)):
        createRecipeFile(distanceToMove[i], i)

if __name__ == "__main__":
    main()


#can request machine to send a screenshot, magnification, xy state ---> THESE ARE THE INPUT
#x y in millimeters (+- 40 millimetters is range)