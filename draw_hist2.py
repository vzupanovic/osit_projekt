import cv2
import numpy as np
import cv
from multiprocessing import Process

class HistogramObject:
	def draw(self, img_obj, frame):
		
		h = np.zeros((300,256,1))
		 
		bins = np.arange(256).reshape(256,1)
		
		hist_item = cv2.calcHist([img_obj],[0],None,[256],[0,256])
		cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
		hist=np.int32(np.around(hist_item))
		pts = np.column_stack((bins,hist))
		cv2.polylines(h,[pts],False,255)
		 
		h=np.flipud(h)
		 
		cv2.imshow('Histogram(eye '+str(frame)+')',h)
		cv2.waitKey(10)
		
#if __name__ == "__main__":
#	while(1):
#		grey = cv2.imread("recordings/eye150.jpg", cv.CV_LOAD_IMAGE_GRAYSCALE)
#		histogram = HistogramObject()
#		histogram.draw(grey)
#		print "govno"
#		grey = cv2.imread("recordings/eye102.jpg", cv.CV_LOAD_IMAGE_GRAYSCALE)
#		histogram.draw(grey)
		
