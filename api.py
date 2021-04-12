import os
from fastapi import FastAPI, Request, UploadFile, File, Form, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import uuid
from motor.motor_asyncio import AsyncIOMotorClient
from tqdm import tqdm
from functions import find_face_similarity, ExtractFace, find_face_similarity_test
from deepface.commons import functions
from deepface import DeepFace
import shutil
from fastapi.encoders import jsonable_encoder
from fastapi.openapi.utils import get_openapi

app = FastAPI()


def custom_openapi():
    # if app.openapi_schema:
    #     return app.openapi_schema
    openapi_schema = get_openapi(
        title="POC Testing",
        version="0",
        description="",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url":
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSG8-8hMTgdX4E7qFZGbsLau2iTiC4CKB2ftQ&usqp=CAU"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


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
    # dateset_path = f'tmp/create_{user_id}'

    # if not os.path.exists(dateset_path):
    #     os.makedirs(dateset_path)

    __, embeddings = ExtractFace.extract_video(video_path=video_path,
                                               model=app.face_model,
                                               need_embedding=True,
                                               skip=4,
                                               is_store=False,
                                               only_face=True)

    __, image_embeddings = ExtractFace.extract_face_of_image(
        image_path=image_path,
        need_embedding=True,
        # output_dir=dateset_path,
        model=app.face_model,
        is_store=False,
    )

    embeddings += image_embeddings

    instances = []
    for embedding in embeddings:
        # embedding = app.face_model.predict(face)[0]
        instance = jsonable_encoder(
            PeopleImageModel(user_id=user_id,
                             embedding_image=embedding.tolist()))
        instances.append(instance)

    instances = await request.app.mongodb["deepface"].insert_many(instances)

    # instance = await request.app.mongodb["deepface"].find_one(
    #     {'_id': instances.inserted_ids[0]})

    # content = {"status": 1, **PeopleImageReadModel(**instance).dict()}
    content = {"status": 1}

    os.remove(video_path)
    os.remove(image_path)
    # shutil.rmtree(dateset_path, ignore_errors=True)

    return JSONResponse(status_code=status.HTTP_201_CREATED, content=content)


@app.post("/find-people/")
async def list_tasks(request: Request, image: UploadFile = File(...)):

    image_path = save_upload_file_tmp(image)
    # dateset_path = f"tmp/find_{uuid.uuid4()}"

    __, embeddings = ExtractFace.extract_face_of_image(
        image_path=image_path,
        need_embedding=True,
        # output_dir=dateset_path,
        model=app.face_model,
        is_store=False,
    )
    # embedding = app.face_model.predict(faces[0])[0]

    pipeline = [{
        "$addFields": {
            "target_embedding": embeddings[0].tolist()
        }
    }, {
        "$unwind": {
            "path": "$embedding_image",
            "includeArrayIndex": "embedding_index"
        }
    }, {
        "$unwind": {
            "path": "$target_embedding",
            "includeArrayIndex": "target_index"
        }
    }, {
        "$project": {
            "user_id": 1,
            "embedding_image": 1,
            "target_embedding": 1,
            "compare": {
                "$cmp": ['$embedding_index', '$target_index']
            }
        }
    }, {
        "$match": {
            "compare": 0
        }
    }, {
        "$group": {
            "_id": {
                "_id": "$_id",
                "user_id": "$user_id"
            },
            "distance": {
                "$sum": {
                    "$pow": [{
                        "$subtract": ['$embedding_image', '$target_embedding']
                    }, 2]
                }
            }
        }
    }, {
        "$project": {
            "_id": 1,
            "distance": {
                "$sqrt": "$distance"
            }
        }
    }, {
        "$project": {
            "_id": 1,
            "distance": 1,
            "cond": {
                "$lte": ["$distance", 10]
            }
        },
    }, {
        "$match": {
            "cond": True
        }
    }, {
        "$sort": {
            "distance": 1
        }
    }, {
        "$limit": 10
    }, {
        "$group": {
            "_id": {
                "user_id": "$_id.user_id",
                "cond": "$cond"
            },
            "distance": {
                "$avg": {
                    "$cond": [{
                        "$lte": ["$distance", 10]
                    }, "$distance", 0]
                }
            }
        }
    }]

    data = {}
    async for doc in request.app.mongodb["deepface"].aggregate(pipeline):
        print(f"doc : {doc}")
        data = {
            "user_id": doc['_id']['user_id'],
            "similarity_avg": (20 - doc['distance']) / 20,
        }

    content = {"status": 1, **data}

    os.remove(image_path)
    return JSONResponse(status_code=status.HTTP_200_OK, content=content)


@app.on_event("startup")
async def startup_db_client():
    app.mongodb_client = AsyncIOMotorClient('mongodb://root:root@mongo')
    app.mongodb = app.mongodb_client['deepface']
    app.face_model = DeepFace.build_model("Facenet")
    # start client here and reuse in future requests


@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()

    # stop your client here
