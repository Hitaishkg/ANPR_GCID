import pytesseract
from PIL import Image

# Loading the image
img = Image.open('test.png')

# Applying OCR using Tesseract OCR engine
text = pytesseract.image_to_string(img)

# Printing the recognized text
print(text)