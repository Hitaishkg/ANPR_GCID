import os, io
from google.cloud import vision
import pandas as pd

file_name = "lisence_plate5.jpg"

image_path = f"/home/vishnu/Documents/ANMR_GCID/model/images/{file_name}"
def google_ocr(img_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"/home/vishnu/Documents/ANMR_GCID/model/vision-385506-f15d7f5ffc05.json"
    client = vision.ImageAnnotatorClient()
    with io.open(img_path, "rb") as image_file:
        content = image_file.read()

    # construct an iamge instance
    image = vision.Image(content=content)

    """
    # or we can pass the image url
    image = vision.types.Image()
    image.source.image_uri = 'https://edu.pngfacts.com/uploads/1/1/3/2/11320972/grade-10-english_orig.png'
    """

    # annotate Image Response
    response = client.text_detection(image=image)  # returns TextAnnotation
    df = pd.DataFrame(columns=["locale", "description"])

    texts = response.text_annotations
    for text in texts:
        df = df.append(
            dict(locale=text.locale, description=text.description), ignore_index=True
        )

    print(df["description"][0])
    return df["description"][0]
google_ocr(image_path)