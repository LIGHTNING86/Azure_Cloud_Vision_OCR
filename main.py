from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import PlainTextResponse, RedirectResponse
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image
import io
import time
import cv2

# Set up Azure Computer Vision credentials
subscription_key = '<your_subscription_key>'
endpoint = '<your_endpoint_link>'

# Initialize the FastAPI app
app = FastAPI()

# Root endpoint to redirect to /docs
@app.get("/", include_in_schema=False)
def read_root():
    return RedirectResponse(url="/docs")

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

@app.post("/predict")
async def predict(file: UploadFile = File(None), use_camera: bool = Form(False)):
    if use_camera:
        # Capture frame from the camera
        capture = cv2.VideoCapture(0)
        ret, frame = capture.read()
        capture.release()
        if not ret:
            return PlainTextResponse(content="Failed to capture image from camera", status_code=400)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        if file is None:
            return PlainTextResponse(content="No file uploaded", status_code=400)
        # Read the uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    contents = buf.getvalue()

    extracted_text = extract_text(contents)

    # Return the extracted text as plain text
    return PlainTextResponse(content=extracted_text)
