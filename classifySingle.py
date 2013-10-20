import cv
import cv2
import numpy

class Detekcija_ociju:
	
	def loadImage(self, img_src):
		grey = cv2.imread(img_src, cv.CV_LOAD_IMAGE_GRAYSCALE)
		cv2.normalize(grey, grey, 0, 255, cv.CV_MINMAX)
		return grey
		
	def normalizeImage(self, img):
		cv.Normalize(img, img, 0, 255, cv.CV_MINMAX)
		return img
		
	def calculateHistogram(self, img_array, bucket):
		hist = cv2.calcHist([img_array],[0],None,[bucket],[0,256])
		return hist	
		
	def loadClassifier(self, slika):
		haarFace = cv.Load('haarcascade_frontalface_default.xml')
		haarEyes = cv.Load('haarcascade_eye.xml')
		storage = cv.CreateMemStorage()
		detectedFace = cv.HaarDetectObjects(slika, haarFace, storage)
		detectedEyes = cv.HaarDetectObjects(slika, haarEyes, storage)
		return detectedFace, detectedEyes
				
	def getFeatures(self, max_count):
		counter = 0
		for i in range(0, max_count):
			image = cv2.imread("cropped_eyes/img"+str(i)+".jpg")
			print "HIST", self.calculateHistogram(image, 16)
				
	def showImage(self, slika):
		cv2.imshow("bla",slika)
		cv.WaitKey()
		
	def trainClassifier(self):
		features = []
		classes = []
		train_source = open('train_set.txt','r')
		train_data = train_source.readlines()
		for line in train_data:
			line_data = []
			line = line.strip()
			line_data = line.split(" ")
			im_class = line_data[1]
			img = line_data[0]
			classes.append(im_class)
			feature = self.calculateHistogram(self.loadImage(img),32);
			features.append(feature)
		features = numpy.matrix(numpy.array(features)).astype("float32")
		classes = numpy.array(numpy.array(classes)).astype("int32")
		classifier = cv2.KNearest()
		classifier.train(features, classes)
		return classifier
		
	def applyClassifier(self, classifier, inputFeatures):
		result = classifier.find_nearest(inputFeatures, 1)
		return result
		
	def classifySingle(self, imgName, classifier):
		self.class_names = {1 : 'poluotvoreno', 2 : 'normalno', 3 : 'turbo', 4 : 'zatvoreno'}
		feature = [self.calculateHistogram(self.loadImage(imgName),32)];
		feature =  numpy.matrix(numpy.array(feature)).astype("float32")
		_, result, _, _ = self.applyClassifier(classifier, feature)
		return self.class_names[int(result)]
				
		
if __name__ == "__main__":
	det_oci = Detekcija_ociju()
	classifier = det_oci.trainClassifier()
	det_oci.classifySingle('recordings_classify/eye131.jpg',classifier)

	
	
	
