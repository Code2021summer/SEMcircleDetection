# has to take images and stitch them together
import cv2 as cv
from pathlib import Path
from PIL import Image
#img = Image.open()
#status of 1 from stitcher means there wasn't enough overlap
#ideas: make stitcher

def main():
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


if __name__ == '__main__':
    main()

#img = Image.open("SquigglesFull.jpg")
#img.show()
#affine based model can be used for specialized devices