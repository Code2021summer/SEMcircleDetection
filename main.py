from scrap import * 

def main():
    img = cv.imread("SEMImgTest/testReal.bmp") #can be changed to image of your choice
    coords_of_center = (30, 20) #adjust as needed
    numberOfPointsRequested = 6 #adjust as needed
    
    rep_points, scale = findRepPoints(img)
    constants = solveCircle(rep_points)
    pixel_points_of_circle = createCircle(constants, numberOfPointsRequested)
    real_points = []
    for coordPair in pixel_points_of_circle:
        real_coord = convertPixelToReal(coordPair, scale, coords_of_center)
        real_points.append(real_coord)
    for i in range(len(real_points)):
        createRecipeFile(real_points[i], i)



if __name__ == '__main__':
    main()
