import cv2
import requests
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from io import BytesIO

#------------------------------------------------------------
# Querying custom vision to extract only the barcodes



# Now there is a trained endpoint that can be used to make a prediction

prediction_key = "<YOUR PREDICTION KEY>"
ENDPOINT = 'https://westeurope.api.cognitive.microsoft.com/'

base_image_url = "./"
image_relative_path = "barcode-on-food-item-c0bf0m.jpg"

predictor = CustomVisionPredictionClient(prediction_key, endpoint=ENDPOINT)

# Open the sample image and get back the prediction results.
with open(base_image_url + "images/" + image_relative_path, mode="rb") as test_data:
    results = predictor.detect_image('<CUSTOM_VISION_AI_PROJECT_ID>', 'Iteration1', test_data)


#------------------------------------------------------------
# Selecting the best candidates and storing the temporal 
# image

class DetectedObject(object):
    
    def __init__(self):
        self.probability = 0.0
        self.bounding_box_left = 0.0
        self.bounding_box_top = 0.0
        self.bounding_box_width = 0.0
        self.bounding_box_height = 0.0

# Display the results.    
max_prob = 0
best_candidate = DetectedObject()

for prediction in results.predictions:
    print("\t" + prediction.tag_name + ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".\
         format(prediction.probability * 100, prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height))
    if max_prob < prediction.probability:
        max_prob = prediction.probability
        best_candidate.probability = prediction.probability
        best_candidate.bounding_box_left = prediction.bounding_box.left
        best_candidate.bounding_box_top = prediction.bounding_box.top
        best_candidate.bounding_box_width = prediction.bounding_box.width
        best_candidate.bounding_box_height = prediction.bounding_box.height

img = cv2.imread(base_image_url + "images/" + image_relative_path)
height, width, depth = img.shape

y0 = int(height * best_candidate.bounding_box_top)
y1 = int(height * (best_candidate.bounding_box_top + best_candidate.bounding_box_height))
dy = y1 - y0
ddy = dy * .2
x0 = int(width * best_candidate.bounding_box_left)
x1 = int(width * (best_candidate.bounding_box_left + best_candidate.bounding_box_width))
dx = x1 - x0
ddx = dx * .2   

yy0 = int(max([y0 - ddy,0]))
yy1 = int(min([y1 + ddy, width]))
xx0 = int(max([x0 - ddx,0]))
xx1 = int(min([x1 +ddx, height]))

crop_img = img[yy0:yy1, \
               xx0:xx1]

cv2.imwrite('tmp.png',crop_img)

#------------------------------------------------------------
# Parsing image with OCR CV API

cv_subscription_key = "<COMPUTER VISION SUBSCRIPTION KEY>"
cv_endpoint = 'https://ocrcv.cognitiveservices.azure.com/'
ocr_url = cv_endpoint + "vision/v2.1/ocr"
#ocr_url = cv_endpoint + "vision/v2.1/read"

# Additional parameters to be used by the API
params = {'language': 'unk', 'detectOrientation': 'true'}

image_path = base_image_url + "tmp.png"
# Read the image into a byte array
image_data = open(image_path, "rb").read()
# Set Content-Type to octet-stream
headers = {'Ocp-Apim-Subscription-Key': cv_subscription_key, 'Content-Type': 'application/octet-stream'}
# put the byte array into your post request
response = requests.post(ocr_url, headers=headers, params=params, data = image_data)
response.raise_for_status()

analysis = response.json()

# Extract the word bounding boxes and text.
line_infos = [region["lines"] for region in analysis["regions"]]
word_infos = []
for line in line_infos:
    for word_metadata in line:
        for word_info in word_metadata["words"]:
            word_infos.append(word_info)
print(word_infos)



# Display the image and overlay it with the extracted text.
plt.figure(figsize=(5, 5))
image =  cv2.imread('tmp.png')
ax = plt.imshow(image, alpha=0.5)
for word in word_infos:
    bbox = [int(num) for num in word["boundingBox"].split(",")]
    text = word["text"]
    origin = (bbox[0], bbox[1])
    patch = Rectangle(origin, bbox[2], bbox[3],
                      fill=False, linewidth=2, color='r')
    ax.axes.add_patch(patch)
    plt.text(origin[0], origin[1], text, fontsize=20, weight="bold", va="top")
plt.axis("off")