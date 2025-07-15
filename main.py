# has to take images and stitch them together
import cv2 as cv
from PIL import Image
#img = Image.open()
#status of 1 from stitcher means there wasn't enough overlap
#ideas: make stitcher
def main():
    print("Wassup")
    imgs = [cv.imread("SquigglesLTest.jpg"), cv.imread("SquigglesRTest.jpg")]
    stitcher = cv.Stitcher.create()
    (status, stitched) = stitcher.stitch(imgs)

    if status == cv.Stitcher_OK:
        cv.imwrite("TestResult.jpg", stitched)
    else:
        print(f"Bruh: {status}")


if __name__ == '__main__':
    main()

#img = Image.open("SquigglesFull.jpg")
#img.show()
#affine based model can be used for specialized devices