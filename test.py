from functions import find_face_similarity, ExtractFace, find_face_similarity_test, test
from deepface.commons import functions
from deepface import DeepFace
import timeit
import os

start = timeit.default_timer()

video_path = 'test_data/5s.mp4'
image_path = 'test_data/test1.png'
dateset_path = 'dataset'

if not os.path.exists(dateset_path):
    os.makedirs(dateset_path)

# deeptest.find_face_similarity_test(video_path=video_path,
#                                    image_path=image_path,
#                                    dateset_path=dateset_path)

# deeptest.find_face_similarity(video_path=video_path,
#                               image_path=image_path,
#                               dateset_path=dateset_path)

model = DeepFace.build_model("Facenet")

faces = test(video_path=video_path,
             output_dir=dateset_path,
             model=model,
             skip=4,
             is_store=True,
             only_face=True)

stop = timeit.default_timer()

print('Time: ', stop - start)
