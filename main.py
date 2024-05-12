from http import HTTPStatus

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from io import BytesIO
from base64 import b64decode
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')


def predict(b64_img):
    image = Image.open(BytesIO(b64decode(b64_img))).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


class Data(BaseModel):
    image: str


@app.post("/get-text", status_code=HTTPStatus.OK)
async def get_text(data: Data):
    return {"text": predict(data.dict()['image'])}
