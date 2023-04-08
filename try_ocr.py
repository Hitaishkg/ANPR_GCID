import pytesseract
from PIL import Image

# Loading the image
# pytesseract.pytesseract.tesseract_cmd = r'/usr/share/tesseract-ocr/4.00/tessdata' #CHANGE THIS BEFORE COMMIT
img = Image.open('/home/vishnu/Documents/ANMR_GCID/finalimg.jpeg')    
# pytesseract.pytesseract.tesseract_cmd = 'C:/OCR/Tesseract-OCR/tesseract.exe'  # your path may be different

# Applying OCR using Tesseract OCR engine
text = pytesseract.image_to_string(img)

# Printing the recognized text
print(text)