# import the necessary packages
from imutils.video import VideoStream
from playsound import playsound
from textblob import TextBlob
import speech_recognition as sr
from pyzbar import pyzbar
from gtts import gTTS
import trafilatura
import imutils
import json
import time
import cv2

# speech to text
sample_rate = 48000
chunk_size = 2048
text = ''

# open API key for Google Cloud Speech API
with open('speech-to-text-key.json', "r") as file:
    j  = file.read()

r = sr.Recognizer()

with sr.Microphone() as source:

	r.adjust_for_ambient_noise(source)
	print("Say Something")

	audio = r.listen(source)
		
	try:
		text = r.recognize_google_cloud(audio, credentials_json=j)
		print(text)

	except sr.UnknownValueError:
		print("Google Speech Recognition could not understand audio")

	except sr.RequestError as e:
		print("Could not request results from Google Speech Recognition service; {0}".format(e))

# if command matches, start video stream
if str(text).strip() == 'barcode' or str(text).strip() == 'scan barcode':

	# initialize the video stream and allow the camera sensor to warm up
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

	barcodeData = None
	detected = False

	# loop over the frames from the video stream
	while not detected:

		# grab the frame from the threaded video stream and resize it to
		# have a maximum width of 400 pixels
		frame = vs.read()
		frame = imutils.resize(frame, width=400)

		# find the barcodes in the frame and decode each of the barcodes
		barcodes = pyzbar.decode(frame)

		# loop over the detected barcodes
		for barcode in barcodes:

			# extract the bounding box location of the barcode and draw
			# the bounding box surrounding the barcode on the image
			(x, y, w, h) = barcode.rect
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
		
			# the barcode data is a bytes object so if we want to draw it
			# on our output image we need to convert it to a string first
			barcodeData = barcode.data.decode("utf-8")
			barcodeType = barcode.type

			# draw the barcode data and barcode type on the image
			text = "{} ({})".format(barcodeData, barcodeType)
			cv2.putText(frame, text, (x, y - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		
			detected = True		

		# show the output frame
		cv2.imshow("Barcode Scanner", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# cleanup
	print("[INFO] cleaning up...")
	cv2.destroyAllWindows()
	vs.stop()

	print(barcodeData)

	# google search the barcode
	print("[INFO] Search...")

	try:
		from googlesearch import search
	except ImportError:
		print("[INFO] No module named 'google' found")

	searche_results = search(barcodeData, num=10, stop=10, pause=2)

	# fetch text from url
	print("[INFO] Fetch...")

	while True:
		try:
			url = next(searche_results)
			page = trafilatura.fetch_url(url)
			result = trafilatura.extract(page, favor_precision=True)
			print(url)
		except TypeError:
			pass
		else:
			print(result)
			break

	# text to speech
	tts = gTTS(result, lang='ru')
	tts.save('result.mp3')

	playsound('result.mp3')
else:
	print("Command not recognized.")