import pytesseract as tess
from PIL import Image


img = Image.open('/mnt/32F6E6CAF6E68D83/kaam/GCID_ANPR/Datasets/img1.png')
text = tess.image_to_string(img)

print(text)
