import numpy as np
from fastapi import FastAPI, File, UploadFile, Body
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

MODEL_POTATO = tf.keras.models.load_model("../saved_model/3")
MODEL_PEPPER = tf.keras.models.load_model("../saved_modelPepper/2")
MODEL_TOMATO = tf.keras.models.load_model("../saved_modelTomato/2")

CLASS_NAMES_POTATO = ["Early Blight", "Late Blight", "Healthy"]
CLASS_NAMES_PEPPER = ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy"]
CLASS_NAMES_TOMATO = ['Tomato_Bacterial_spot',
                      'Tomato_Early_blight',
                      'Tomato_Late_blight',
                      'Tomato_Leaf_Mold',
                      'Tomato_Septoria_leaf_spot',
                      'Tomato_Spider_mites_Two_spotted_spider_mite',
                      'Tomato__Target_Spot',
                      'Tomato__Tomato_YellowLeaf__Curl_Virus',
                      'Tomato__Tomato_mosaic_virus',
                      'Tomato_healthy']


@app.get("/ping")
async def ping():
    return "Hello"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())

    img_batch = np.expand_dims(image, 0)

    prediction = MODEL_POTATO.predict(img_batch)

    predicted_class = CLASS_NAMES_POTATO[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


# @app.post("/predict2")
# async def predict2(
#         file: UploadFile = File(...)
# ):
#     image2 = read_file_as_image(await file.read())
#
#     img_batch = np.expand_dims(image2, 0)
#
#     prediction = MODEL_PEPPER.predict(img_batch)
#
#     predicted_class = CLASS_NAMES_PEPPER[np.argmax(prediction[0])]
#     confidence = np.max(prediction[0])
#
#     return {
#         'class': predicted_class,
#         'confidence': float(confidence)
#     }
#
#
# @app.post("/predict3")
# async def predict3(
#         file: UploadFile = File(...)
# ):
#     image3 = read_file_as_image(await file.read())
#
#     img_batch3 = np.expand_dims(image3, 0)
#
#     prediction3 = MODEL_TOMATO.predict(img_batch3)
#
#     predicted_class3 = CLASS_NAMES_TOMATO[np.argmax(prediction3[0])]
#     confidence3 = np.max(prediction3[0])
#
#     return {
#         'class': predicted_class3,
#         'confidence': float(confidence3)
#     }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
