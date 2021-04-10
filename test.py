import functions as deeptest
import timeit
import os

start = timeit.default_timer()

video_path = 'test_data/5s.mp4'
image_path = 'test_data/test1.png'
dateset_path = 'tmp/validate_dataset'

if not os.path.exists(dateset_path):
    os.makedirs(dateset_path)

deeptest.find_face_similarity_test(video_path=video_path,
                                   image_path=image_path,
                                   dateset_path=dateset_path)

# deeptest.find_face_similarity(video_path=video_path,
#                               image_path=image_path,
#                               dateset_path=dateset_path)

stop = timeit.default_timer()

print('Time: ', stop - start)
