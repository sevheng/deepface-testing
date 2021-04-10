import os
from fastapi import FastAPI, Request, UploadFile, File, Form, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from motor.motor_asyncio import AsyncIOMotorClient
from tqdm import tqdm
from functions import find_face_similarity, ExtractFace, find_face_similarity_test
from deepface.commons import functions
from deepface import DeepFace
import shutil
from fastapi.encoders import jsonable_encoder

app = FastAPI()


class PeopleImageModel(BaseModel):
    id: str = Field(default_factory=uuid.uuid4, alias="_id")
    user_id: int
    embedding_image: list

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "id": "00010203-0405-0607-0809-0a0b0c0d0e0f",
                "user_id": 2,
                "embedding_image": [],
            }
        }


class PeopleImageReadModel(BaseModel):
    id: str = Field(default_factory=uuid.uuid4, alias="_id")
    user_id: int

    class Config:
        schema_extra = {
            "example": {
                "id": "00010203-0405-0607-0809-0a0b0c0d0e0f",
                "user_id": 2,
            }
        }


def save_upload_file_tmp(upload_file: UploadFile):
    extension = upload_file.filename.split('.')[1]
    file_location = f"tmp/{uuid.uuid4()}.{extension}"
    with open(file_location, "wb+") as file_object:
        file_object.write(upload_file.file.read())
    return file_location


@app.post("/validate/")
async def validate(request: Request,
                   image: UploadFile = File(...),
                   video: UploadFile = File(...)):

    video_path = save_upload_file_tmp(video)
    image_path = save_upload_file_tmp(image)
    dateset_path = f'tmp/validate_dataset_{uuid.uuid4()}'

    if not os.path.exists(dateset_path):
        os.makedirs(dateset_path)

    similarity, blurry = find_face_similarity(video_path=video_path,
                                              image_path=image_path,
                                              dateset_path=dateset_path)

    os.remove(video_path)
    os.remove(image_path)
    os.rmdir(dateset_path)

    print(f"Video is blurry: {blurry}")
    if similarity:
        content = {"status": 1, **similarity}
    else:
        content = {
            "status": 0,
            "message": "Image is not similar with video frame."
        }
        if blurry:
            content['message'] = "Most of video frame are blurry."

    return JSONResponse(status_code=status.HTTP_200_OK, content=content)


@app.post("/validate2/")
async def validate2(request: Request,
                    image: UploadFile = File(...),
                    video: UploadFile = File(...)):

    video_path = save_upload_file_tmp(video)
    image_path = save_upload_file_tmp(image)
    dateset_path = f'tmp/validate_dataset_{uuid.uuid4()}'

    if not os.path.exists(dateset_path):
        os.makedirs(dateset_path)

    similarity, blurry = find_face_similarity_test(video_path=video_path,
                                                   image_path=image_path,
                                                   dateset_path=dateset_path)

    os.remove(video_path)
    os.remove(image_path)
    os.rmdir(dateset_path)

    print(f"Video is blurry: {blurry}")
    if similarity:
        content = {"status": 1, **similarity}
    else:
        content = {
            "status": 0,
            "message": "Image is not similar with video frame."
        }
        if blurry:
            content['message'] = "Most of video frame are blurry."

    return JSONResponse(status_code=status.HTTP_200_OK, content=content)


@app.post("/create-people/")
async def create_people(request: Request,
                        user_id: int = Form(...),
                        image: UploadFile = File(...),
                        video: UploadFile = File(...)):

    video_path = save_upload_file_tmp(video)
    image_path = save_upload_file_tmp(image)
    dateset_path = f'tmp/create_{user_id}'

    if not os.path.exists(dateset_path):
        os.makedirs(dateset_path)

    ExtractFace.extract(video_path=video_path, output_dir=dateset_path)

    facial_img_paths = [image_path]
    for root, directory, files in os.walk(dateset_path):
        for file in files:
            if '.jpg' in file:
                facial_img_paths.append(root + "/" + file)

    model = DeepFace.build_model("Facenet")
    instances = []
    for i in tqdm(range(0, len(facial_img_paths))):
        facial_img_path = facial_img_paths[i]
        facial_img = functions.preprocess_face(facial_img_path,
                                               target_size=(160, 160),
                                               detector_backend='ssd',
                                               enforce_detection=False)
        embedding = model.predict(facial_img)[0]

        instances.append(
            jsonable_encoder(
                PeopleImageModel(user_id=user_id,
                                 embedding_image=embedding.tolist())))

    instances = await request.app.mongodb["deepface"].insert_many(instances)
    instance = await request.app.mongodb["deepface"].find_one(
        {'_id': instances.inserted_ids[0]})

    content = {"status": 1, **PeopleImageReadModel(**instance).dict()}

    os.remove(video_path)
    os.remove(image_path)
    shutil.rmtree(dateset_path, ignore_errors=True)

    return JSONResponse(status_code=status.HTTP_201_CREATED, content=content)


@app.get("/find-people/")
async def list_tasks(request: Request, image: UploadFile = File(...)):
    content = {}
    return JSONResponse(status_code=status.HTTP_200_OK, content=content)


@app.on_event("startup")
async def startup_db_client():
    app.mongodb_client = AsyncIOMotorClient('mongodb://root:root@mongo')
    app.mongodb = app.mongodb_client['deepface']
    # start client here and reuse in future requests


@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()

    # stop your client here
