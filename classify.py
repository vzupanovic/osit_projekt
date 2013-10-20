import cv
import cv2
import numpy
import classifySingle
from os import listdir

class ApplyClassifierMulti:
	def Classify(self, folderLocation):
		badCount = 0
		missCount = 0
		totalCount = 0
		det_oci = classifySingle.Detekcija_ociju()
		classifier = det_oci.trainClassifier()
		for image in listdir('recordings_classify/'):
			result = det_oci.classifySingle('recordings_classify/'+image,classifier)
			print image, result
			
if __name__ == "__main__":
	test_classifier = ApplyClassifierMulti()
	test_classifier.Classify('recordnigs_classify')
	
