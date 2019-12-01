from skimage.measure import compare_ssim
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
                help = "first input image")
ap.add_argument("-s", "--second", required=True,
                help="second")
args = vars(ap.parse_args())

#loading two input images
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])

#converting images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print(f"SSIM: {score}")

thresh = cv2.threshold(diff, 0, 255,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for c in cnts:
    (x,y,w,h) = cv2.boundingRect(c)
    cv2.rectangle(imageA, (x,y), (x+w, y+h), (0,0,255), 2)
    cv2.rectangle(imageB, (x,y), (x+w, y+h), (0,0,255), 2)


cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)

cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)

cv2.waitKey(0)

