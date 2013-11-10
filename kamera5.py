import cv2

class WebCamSream:
	def loadStream(self, downscale):
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
				cropped1 = cv2.resize(cropped1,(80,50))
				cropped2 = cv2.resize(cropped2,(80,50))
				cv2.imwrite('recordings_test1/eye_1_'+str(detected)+'.jpg',cropped1)
				cv2.imwrite('recordings_test1/eye_2_'+str(detected)+'.jpg',cropped2)
				cv2.rectangle(frame, (x,y-h/8), (x+w/2-w/8,y-h/8+h), (0,255,0))
				cv2.rectangle(frame, (x + w/2+w/8,y-h/8), (x+w,y-h/8+h), (255,0,0))
				detected = detected + 1
		 
			cv2.putText(frame, "Press ESC to close.", (5, 25),
			cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
			cv2.imshow("preview", frame)
		 
			# get next frame
			rval, frame = webcam.read()
		 
			key = cv2.waitKey(20)
			if key in [27, ord('Q'), ord('q')]: # exit on ESC
				break

if __name__ == "__main__":
	stream = WebCamSream()
	stream.loadStream(2)		
