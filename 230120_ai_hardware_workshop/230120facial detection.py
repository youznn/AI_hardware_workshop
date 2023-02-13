#카메라 예제  deep learning 모델 사용하기
import sys
import cv2 #처음부터 프로젝트 할 때는 이 라이브러리를 따로 미리 설치해야함
from tflite_support.task import core, vision
import board, adafruit_ssd1306 
from PIL import Image, ImageDraw, ImageFont
import time
from grove.grove_servo import GroveServo
from grove.grove_button import GroveButton

PIN =5
servo = GroveServo(24)
button = GroveButton(PIN)

def classify(model, labels, image):
	classifier_options = vision.ImageClassifierOptions(base_options = core.BaseOptions(file_name=model)) #관용구라고 생각하면 편함
	classifier = vision.ImageClassifier.create_from_options(classifier_options) #인스턴스 만들기
	
	rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #BGR을 RGB 로 바꾸기
	tensor_image = vision.TensorImage.create_from_array(rgb_image) #tensor 이미지로 바꾸기
	
	result = classifier.classify(tensor_image) #[[분류1], [분류2], [분류3]...] [분류1] 안에는 카테고리 별 확률[카테고리1, 카테고리2, ...]
	category = result.classifications[0].categories[0] 
	category_name = labels[category.index] #우리가 정한 label 순서/
	category_probability = round(category.score,2)
	
	return (category_name, category_probability)
	
	
def dp(_animal):
	WIDTH=128
	HEIGHT = 64

	i2c = board.I2C()
	oled = adafruit_ssd1306.SSD1306_I2C(WIDTH, HEIGHT, i2c, addr=0X3C) 

	oled.fill(0)
	oled.show()
	#동물 별 이미지 데이터 만들어야함, 딕셔너리로 가자 완!
	animal_image = {'0 cat\n':'./happycat_oled_64.ppm', '1 dog\n':'./dog.png'}
	
	image= Image.open(animal_image[_animal]).convert('1') #convert(1)은 컬러 모드에서 흑백
	image = image.transpose(Image.FLIP_TOP_BOTTOM)
	oled.image(image)
	
	oled.show()
	
	image = Image.new('1', (WIDTH, HEIGHT), 0) #없애야 할 것 같음;;
	brush = ImageDraw.Draw(image)

	font = ImageFont.truetype('malgun.ttf', 10) #뒤에 숫자는 폰트 크기
	if _animal == '0 cat\n':
		brush.text((0,20), _animal, font = font, fill = 1)
		servo.setAngle(170)

	elif _animal == '1 dog\n':
		brush.text((0,20), _animal, font = font, fill = 0)
		servo.setAngle(10)

	
	

	oled.image(image) #붓으로 그린 것을 메모리에 올리는 명령어
	oled.show() #메모리에 올려놓은 그림을 화면에 표시하는 명령어
	

	
	
def main():
	#모델과 레이블 읽어들이기
	model = './model.tflite' #얼굴나이 모델
	
	f = open('./labels.txt', 'r') #얼굴 나이 라벨 txt
	labels = f.readlines()
	f.close()
	
	#video 캡쳐 시작
	cap=cv2.VideoCapture(0) #인스턴스 만들기
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224) #가로 640px
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224) #세로 480px

	#영상캡쳐, while 구문 동안 카메라 장면들이 계속 캡쳐 돼서 화면에 표시되고, 특정 키보드를 눌렀을 때, 그 시점의
	#image가 저장된다.
	while cap.isOpened():
		success, image = cap.read() #한 장면이 캡쳐
		if not success:
			sys.exit('ERROR')
		
		image = cv2.flip(image, 1)
		#image = ('~/Desktop/selfie.jpg')
		
		prediction = classify(model, labels, image) #튜플 형식
		text = prediction[0] +':' + str(prediction[1]*100)+'%'
		
		_FONT_SIZE = 1
		_FONT_THICKNESS = 1
		_FONT = cv2.FONT_HERSHEY_PLAIN
		
		#cv2.flip(image, 1) # 0 : 상하반전, 1 : 좌우반전, -1: 상하좌우반전
		
		#기능1 화면에 '동물 : 몇% ' 띄우기
		cv2.putText(image, text, (10, 200),_FONT, _FONT_SIZE,\
		(0,0,255),_FONT_THICKNESS) #색깔 순서 BGR, 마지막 숫자 1은 글씨 굵기
		#폰트 :cv2.FONT_HERSHEY_PLAIN
		
		cv2.imshow('preview', image) #preview라는 화면(window)에 이미지를 그려주는 함수
		
		#기능2 OLED화면에 해당 동물 픽셀 사진 띄우기
		#기능3 OLED화면에 멘트 띄우기
		_animal = prediction[0]

		#마지막 기능 셀카 찍기 (버튼 누르면 셀카 찍기로 바꿔야함)
		#if cv2.waitKey(1) == 65: #대문자 A가 눌렸다면 
		if button:
			cv2.imwrite('selfie.jpg', image) #A눌렀을 때의 image를 저장
		
		dp(_animal)
	
		#ESC 누르면 종료
		if cv2.waitKey(1) == 27:
			break
			
	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
