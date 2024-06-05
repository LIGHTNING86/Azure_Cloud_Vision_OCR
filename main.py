from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image, ImageDraw
import io
import time
import base64

# Set up Azure Computer Vision credentials
subscription_key = '4dcf389a09b9486a9ec866de7d9f4a54'
endpoint = 'https://azure-vision-ai-testing.cognitiveservices.azure.com/'

# Initialize the FastAPI app
app = FastAPI()

# Function to extract text and draw bounding boxes
def extract_text_with_bounding_boxes(image_content):
    # Authenticate the client
    credentials = CognitiveServicesCredentials(subscription_key)
    client = ComputerVisionClient(endpoint, credentials)

    # Call the read operation
    raw_response = client.read_in_stream(io.BytesIO(image_content), language="en", raw=True)

    # Get the operation ID
    operation_location = raw_response.headers["Operation-Location"]
    operation_id = operation_location.split("/")[-1]

    # Wait for the operation to complete
    while True:
        result = client.get_read_result(operation_id)
        if result.status not in [OperationStatusCodes.not_started, OperationStatusCodes.running]:
            break
        time.sleep(1)

    # Open the image for drawing
    image = Image.open(io.BytesIO(image_content))

    # Check if image has orientation metadata and adjust orientation if needed
    if hasattr(image, '_getexif') and image._getexif() is not None:
        exif = dict(image._getexif().items())
        if 274 in exif:
            orientation = exif[274]
            if orientation == 3:
                image = image.transpose(Image.ROTATE_180)
            elif orientation == 6:
                image = image.transpose(Image.ROTATE_270)
            elif orientation == 8:
                image = image.transpose(Image.ROTATE_90)
        else:
            print("Image does not have orientation metadata.")
    else:
        print("Image does not have EXIF metadata.")

    draw = ImageDraw.Draw(image)

    extracted_texts = []

    # Draw bounding boxes on the image
    if result.status == OperationStatusCodes.succeeded:
        for line in result.analyze_result.read_results[0].lines:
            # Extract bounding box coordinates
            left = line.bounding_box[0]
            top = line.bounding_box[1]
            right = line.bounding_box[4]
            bottom = line.bounding_box[5]

            # Draw bounding box on the image
            draw.rectangle([left, top, right, bottom], outline="red", width=4)
            print("Extracted text:", line.text)

            extracted_texts.append(line.text)
    else:
        print("Text extraction operation did not succeed. Status:", result.status)

    # Save the image with bounding boxes to a bytes buffer
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    byte_im = buf.getvalue()

    return byte_im, extracted_texts

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    image_with_boxes, extracted_texts = extract_text_with_bounding_boxes(contents)
    
    # Return the image with bounding boxes
    return StreamingResponse(io.BytesIO(image_with_boxes), media_type="image/jpeg")

@app.get("/image/{image_name}")
async def get_image(image_name: str):
    return StreamingResponse(io.BytesIO(image_name), media_type="image/jpeg")
