from fastapi import FastAPI, File, UploadFile
from fastapi.responses import PlainTextResponse
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image
import io
import time
import cv2
import numpy as np

# Set up Azure Computer Vision credentials
subscription_key = '4dcf389a09b9486a9ec866de7d9f4a54'
endpoint = 'https://azure-vision-ai-testing.cognitiveservices.azure.com/'

# Initialize the FastAPI app
app = FastAPI()

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Azure Cloud Vision OCR API"}

# Function to extract text
def extract_text(image_content):
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

    extracted_texts = []

    # Extract texts from the image
    if result.status == OperationStatusCodes.succeeded:
        for line in result.analyze_result.read_results[0].lines:
            extracted_texts.append(line.text)
    else:
        print("Text extraction operation did not succeed. Status:", result.status)

    return "\n".join(extracted_texts)

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    extracted_text = extract_text(contents)
    
    # Return the extracted text as plain text
    return PlainTextResponse(content=extracted_text)

@app.post("/capture/")
async def capture_image():
    # Capture frame from the camera
    capture = cv2.VideoCapture(0)
    ret, frame = capture.read()
    capture.release()
    if not ret:
        return PlainTextResponse(content="Failed to capture image from camera", status_code=400)
    
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    contents = buf.getvalue()

    extracted_text = extract_text(contents)
    
    # Return the extracted text as plain text
    return PlainTextResponse(content=extracted_text)
