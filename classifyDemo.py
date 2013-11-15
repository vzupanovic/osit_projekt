#	Python prototip naseg projekta konzolna varijanta
#	Iscrtava i histogram i klasificira u realnom
#	vremenu oba oka... kad izlazite van prvo misem
#	kliknite na prozor gdje se streama pa esc...
#	dalje, testiranje valjanosti klasfikatora
#	ne spada u ovaj dio, to ce bit u posebnom fajlu
#	ovo vam je edukativno tu zbog ispita
#	TODO: graficko sucelje, testiranje_clasifi. (to vise necu ja)
#	mozda TODO: detekcija zjenice, jebe ga zatvoreno oko najvise
#	i ono s foldovima (10) je TODO (mozda ja, 1% ja)

import cv
import cv2
import numpy
import draw_hist2

class Detekcija_ociju:
	
	def loadImage(self, img_src):
		grey = cv2.imread(img_src, cv.CV_LOAD_IMAGE_GRAYSCALE)
		cv2.normalize(grey, grey, 0, 255, cv.CV_MINMAX)
		return grey
	
	def getFrame(self, imgObj):
		grey = cv2.cvtColor(imgObj, cv2.COLOR_BGR2GRAY)
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
		train_source = open('bazaSlika/klasirano.txt','r')
		train_data = train_source.readlines()
		for line in train_data:
			line_data = []
			line = line.strip()
			print line
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
		self.class_names = {1 : 'zatvoreno', 2 : 'poluotvoreno', 3 : 'otvoreno', 4 : 'turbo'}
		feature = [self.calculateHistogram(self.loadImage(imgName),32)];
		feature =  numpy.matrix(numpy.array(feature)).astype("float32")
		_, result, _, _ = self.applyClassifier(classifier, feature)
		return self.class_names[int(result)]
		
	def classifySingleFrame(self, imgObj, classifier):
		self.class_names = {1 : 'zatvoreno', 2 : 'poluotvoreno', 3 : 'otvoreno', 4 : 'turbo'}
		feature = [self.calculateHistogram(self.getFrame(imgObj),32)];
		feature =  numpy.matrix(numpy.array(feature)).astype("float32")
		_, result, _, _ = self.applyClassifier(classifier, feature)
		return self.class_names[int(result)]
		
	def classifyMore(self, dat, classifier):
		score = 0
		test_source = open('bazaSlika/klasificiraj.txt','r')
		test_set = test_source.readlines()
		maximum = len(test_set)
		for line in test_set:
			line_data = []
			line = line.strip()
			line_data = line.split(" ")
			im_class = line_data[1]
			img = line_data[0]
			g_class = self.classifySingle(img, classifier)
			print img, g_class, im_class
			if (g_class == self.class_names[int(im_class)]):
				score = score + 1
		print "SCORE: "+str(score)+"/"+str(maximum)

class WebCamSream:
	def loadStream(self, downscale, eyeObject, classifierEyes):
		frame_num = 1
		TRAINSET = "haarcascade_frontalface_default.xml"
		TRAINSET_EYES = "par-mali.xml"
		DOWNSCALE = downscale
		detected = 0
		webcam = cv2.VideoCapture(0)
		cv2.namedWindow("preview")
		classifier = cv2.CascadeClassifier(TRAINSET)
		classifier_eyes = cv2.CascadeClassifier(TRAINSET_EYES)
		 
		 
		if webcam.isOpened(): # try to get the first frame
			rval, frame = webcam.read()
		else:
			rval = False
		 
		while rval:
		 
			# detect faces and eyes and draw bounding boxes
			minisize = (frame.shape[1]/DOWNSCALE,frame.shape[0]/DOWNSCALE)
			miniframe = cv2.resize(frame, minisize)

			faces = classifier.detectMultiScale(miniframe)
			eyes = classifier_eyes.detectMultiScale(miniframe)
			
			for f in faces:
				x, y, w, h = [ v*DOWNSCALE for v in f ]
				cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))
			
			for eye in eyes:
				x, y, w, h = [ v*DOWNSCALE for v in eye ]
				#cropped = frame[y : (y+h), x : (x+w)]
				cropped1 = frame[y : (y+h), x : (x+w/2-w/8)]
				cropped2 = frame[y : (y+h), (x+w/2+w/8) : (x+w)]
				cropped1 = cv2.resize(cropped1,(80,45))
				name1 = eyeObject.classifySingleFrame(cropped1, classifierEyes)
				histogramEye = draw_hist2.HistogramObject()
				histogramEye.draw(eyeObject.getFrame(cropped1),1)
				name2 = eyeObject.classifySingleFrame(cropped1, classifierEyes)
				histogramEye.draw(eyeObject.getFrame(cropped2),2)
				#name = eyeObject.classifySingle('recordings_test1/eye_2_13.jpg',classifierEyes)
				cropped2 = cv2.resize(cropped2,(80,45))
				print '%5s %20s %20s' % (frame_num, name1, name2)
				#cv2.imwrite('recordings_test1/eye_1_'+str(detected)+'.jpg',cropped1)
				#cv2.imwrite('recordings_test1/eye_2_'+str(detected)+'.jpg',cropped2)
				cv2.rectangle(frame, (x,y-h/8), (x+w/2-w/8,y-h/8+h), (0,255,0))
				cv2.rectangle(frame, (x + w/2+w/8,y-h/8), (x+w,y-h/8+h), (255,0,0))
				detected = detected + 1
		 
			cv2.putText(frame, "Press ESC to close.", (5, 25),
			cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
			cv2.imshow("preview", frame)
			frame_num = frame_num + 1
			# get next frame
			rval, frame = webcam.read()
			key = cv2.waitKey(20)
			if key in [27, ord('Q'), ord('q')]: # exit on ESC
				break
			
if __name__ == "__main__":
	eyeObject = Detekcija_ociju()
	print "-"*30,"Loading data set", "-"*30
	classifier = eyeObject.trainClassifier()
	print "-"*30, "Data set end", "-"*30
	print "Connecting to cammera:"
	print "="*60
	print "Frame number\tClass left eye\tClass right eye"
	print "="*60
	cameraObject = WebCamSream()
	cameraObject.loadStream(2,eyeObject, classifier)
