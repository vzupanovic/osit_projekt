import cv2
import numpy as np
import sys

class pupilDetector:
	def detectPupil(self, frame):
		img = cv2.medianBlur(frame,5)
		cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
		circles = cv2.HoughCircles(img,cv2.cv.CV_HOUGH_GRADIENT,1,10,param1=60,param2=30,minRadius=5,maxRadius=30)
		print circles
		if circles != None:
			circles = np.uint16(np.around(circles))
			for i in circles[0,:]:
				cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),1) # draw the outer circle
				cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3) # draw the center of the circle
		else:
			print "no circles"
		cv2.imshow('detected circles',cimg)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		
if __name__ == "__main__":
	det_zjenice = pupilDetector()
	frame = cv2.imread("bazaSlika/train_set/eye_5_276.jpg",0)
	det_zjenice.detectPupil(frame)
