import os
import time

import cv2
import dlib
import imutils
import numpy as np
import pandas as pd
from deepface import DeepFace
from imutils import face_utils
from imutils.video import FileVideoStream
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace, DeepID, DlibWrapper, ArcFace, Boosting
from deepface.commons import functions, realtime, distance as dst


class LivenessDetector:
    # define two constants, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 3

    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    TOTAL = 0

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        ".deepface/weights/shape_predictor_68_face_landmarks.dat")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    def clean(self):
        self.TOTAL = 0
        self.COUNTER = 0

    def eye_aspect_ratio(self, eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # return the eye aspect ratio
        return ear

    def detect_eyeblink_by_frame(self, frame, is_frame_gray=False):

        # detect faces in the grayscale frame
        if is_frame_gray:
            rects = self.detector(frame, 0)
        else:
            rects = self.detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = self.predictor(frame, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < self.EYE_AR_THRESH:
                self.COUNTER += 1

            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                    self.TOTAL += 1
                    return True

                # reset the eye frame counter
                self.COUNTER = 0

        return False


class ExtractFace:

    # protoPath = ".deepface/weights/deploy.prototxt"
    # modelPath = ".deepface/weights/res10_300x300_ssd_iter_140000.caffemodel"
    # net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # open a pointer to the video file stream and initialize the total
    # number of frames read and saved thus far
    read = 0
    saved = 0

    # of frames to skip before applying face detection
    skip = 0

    # minimum probability to filter weak detections
    confidence = 0.5

    output_dir = ""

    frames = []
    embeddings = []

    model = None

    def __init__(self, output_dir, skip=4, model=None):
        self.output_dir = output_dir
        self.skip = skip
        self.model = model

    def clean(self):
        self.read = 0
        self.saved = 0
        self.frames = []
        for f in os.listdir(self.output_dir):
            os.remove(os.path.join(self.output_dir, f))

    def detect_face(self, frame):

        input_shape = (224, 224)
        if self.model:
            input_shape = functions.find_input_shape(self.model)
        #----------------------------------

        face = functions.preprocess_face(img=frame,
                                         target_size=input_shape,
                                         enforce_detection=False,
                                         detector_backend='ssd')

        # plt.show()

        # face = None
        # # grab the frame dimensions and construct a blob from the frame
        # (h, w) = frame.shape[:2]
        # blob = cv2.dnn.blobFromImage(frame, 1.0, (160, 160),
        #                              (104.0, 177.0, 123.0))

        # # pass the blob through the network and obtain the detections and
        # # predictions
        # self.net.setInput(blob)
        # detections = self.net.forward()

        # # ensure at least one face was found
        # if len(detections) > 0:
        #     # we're making the assumption that each image has only ONE
        #     # face, so find the bounding box with the largest probability
        #     i = np.argmax(detections[0, 0, :, 2])
        #     confidence = detections[0, 0, i, 2]

        #     # ensure that the detection with the largest probability also
        #     # means our minimum probability test (thus helping filter out
        #     # weak detections)
        #     if confidence > self.confidence:
        #         # compute the (x, y)-coordinates of the bounding box for
        #         # the face and extract the face ROI
        #         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        #         (startX, startY, endX, endY) = box.astype("int")
        #         face = frame[startY:endY, startX:endX]

        return face

    def extract_face_by_frame(self, frame, need_embedding, is_store=False):
        self.read += 1

        # check to see if we should process this frame
        if self.read % self.skip != 0:
            return

        face = self.detect_face(frame)
        # write the frame to disk
        if is_store:
            p = os.path.sep.join(
                [self.output_dir, "{}.jpg".format(self.saved)])
            cv2.imwrite(p, face)
            # print("[INFO] saved {} to disk".format(p))
        else:
            self.frames.append(face)

        if need_embedding:
            self.embeddings.append(self.model.predict(face)[0, :])

        self.saved += 1
        # plt.imshow(face[0][:, :, ::-1])
        # plt.savefig(f'tmp/figure{self.saved}.png')
        return face

    def extract_by_frame(self, frame, is_store=False):
        self.read += 1

        # check to see if we should process this frame
        if self.read % self.skip != 0:
            return

        if is_store:
            p = os.path.sep.join(
                [self.output_dir, "{}.jpg".format(self.saved)])
            cv2.imwrite(p, frame)
        else:
            self.frames.append(frame)

        self.saved += 1
        return frame
        # print("[INFO] saved {} to disk".format(p))

    @classmethod
    def extract_face_of_image(cls,
                              image_path,
                              need_embedding=False,
                              output_dir='tmp',
                              model=None,
                              is_store=False):
        extract_face = cls(output_dir=output_dir, model=model, skip=1)
        frame = cv2.imread(image_path)
        extract_face.extract_face_by_frame(frame=frame,
                                           is_store=is_store,
                                           need_embedding=need_embedding)
        return extract_face.frames, extract_face.embeddings

    @classmethod
    def extract_video(cls,
                      video_path,
                      output_dir='tmp',
                      model=None,
                      skip=4,
                      is_store=False,
                      only_face=True,
                      need_embedding=False):
        vs = FileVideoStream(video_path).start()
        fileStream = True

        extract_face = cls(output_dir=output_dir, skip=skip, model=model)
        i = 0
        # loop over frames from the video stream
        while True:
            # if this is a file video stream, then we need to check if
            # there any more frames left in the buffer to process
            if fileStream and not vs.more():
                break

            # grab the frame from the threaded video file stream
            frame = vs.read()

            if not isinstance(frame, type(None)):
                frame = imutils.resize(frame, width=360)
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if only_face:
                    extract_face.extract_face_by_frame(frame=frame,
                                                       is_store=is_store,
                                                       need_embedding=True)
                else:
                    extract_face.extract_by_frame(frame=frame,
                                                  is_store=is_store)
                i += 1
            else:
                break
        # do a bit of cleanup
        vs.stop()
        print(f"iter : {i}")
        return extract_face.frames, extract_face.embeddings


def detect_blur_fft(image, size=60, thresh=10, vis=False):
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    # check to see if we are visualizing our output
    if vis:
        # compute the magnitude spectrum of the transform
        magnitude = 20 * np.log(np.abs(fftShift))

        # display the original input image
        (fig, ax) = plt.subplots(
            1,
            2,
        )
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Input")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # display the magnitude image
        ax[1].imshow(magnitude, cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        # show our plots
        plt.show()

    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)

    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return (mean, mean <= thresh)


def find_face_similarity(video_path, image_path, dateset_path):

    vs = FileVideoStream(video_path).start()
    fileStream = True

    # time.sleep(1.0)

    liveness = False

    frame_rate = np.floor(vs.stream.get(5))

    extract_face = ExtractFace(output_dir=dateset_path, skip=4)
    liveness_detector = LivenessDetector()

    blurry_count = 0
    i = 0
    # loop over frames from the video stream
    while True:
        # if this is a file video stream, then we need to check if
        # there any more frames left in the buffer to process
        if fileStream and not vs.more():
            break

        # grab the frame from the threaded video file stream
        frame = vs.read()

        if not isinstance(frame, type(None)):
            frame = imutils.resize(frame, width=360)

            # convert the frame to grayscale and detect blur in it
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            (mean, blurry) = detect_blur_fft(gray,
                                             size=60,
                                             thresh=10,
                                             vis=False)

            # print(f"blurry: {blurry}, mean: {mean}")
            if not blurry:
                extract_face.extract_by_frame(frame=frame, is_store=True)
                if not liveness:
                    liveness = liveness_detector.detect_eyeblink_by_frame(
                        frame=gray, is_frame_gray=True)
            else:
                blurry_count += 1
            i += 1
        else:
            break
        # print(f"frame {frame}")
    # do a bit of cleanup
    vs.stop()
    print(f"liveness : {liveness}")
    print(f"iter : {i}")
    print(f"blurry_count : {blurry_count}")
    result = {}

    if liveness:
        # distance_metric (string): cosine, euclidean, euclidean_l2
        # detector_backend (string): set face detector backend as mtcnn, opencv, ssd or dlib
        # model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib or Ensemble
        df = DeepFace.find(image_path,
                           db_path=dateset_path,
                           model_name='Facenet',
                           distance_metric='euclidean',
                           detector_backend='ssd',
                           enforce_detection=False)

        if len(df) > 20:
            df = df.iloc[:20]
            result = {
                "euclidean_distance": df.iloc[0]['Facenet_euclidean'],
                "euclidean_distance_avg": df['Facenet_euclidean'].mean(),
                "similarity": (20 - df.iloc[0]['Facenet_euclidean']) / 20,
                "similarity_avg": (20 - df['Facenet_euclidean'].mean()) / 20,
            }
            # print(df)
            print(result)

    extract_face.clean()
    liveness_detector.clean()
    return result, (i * 0.9) < blurry_count


def find(
    image_face,
    dateset,
    distance_metric='cosine',
    model=None,
):

    # tic = time.time()

    # img_paths, bulkProcess = functions.initialize_input(img_path)
    # functions.initialize_detector(detector_backend = detector_backend)

    #-------------------------------

    #---------------------------------------
    print("model load")

    metric_names = []
    metric_names.append(distance_metric)

    #---------------------------------------

    #------------------------
    #find representations for db images
    # print("step 1")
    representations = []

    # pbar = tqdm(range(0,len(employees)), desc='Finding representations')

    #for employee in employees:
    i = 0
    for frame in dateset:

        instance = []
        instance.append(i)

        #----------------------------------
        #decide input shape

        # input_shape = functions.find_input_shape(model)
        # input_shape_x = input_shape[0]
        # input_shape_y = input_shape[1]

        # print(f"input_shape: {input_shape}")
        # #----------------------------------

        # img = functions.preprocess_face(img=frame,
        #                                 target_size=(input_shape_y,
        #                                              input_shape_x),
        #                                 enforce_detection=False,
        #                                 detector_backend='ssd')

        # print(f"image: {img.shape} ")
        # print(f"frame {frame.shape} ")

        representation = model.predict(frame)[0, :]
        instance.append(representation)

        #-------------------------------

        representations.append(instance)
        i += 1

    # f = open(db_path+'/'+file_name, "wb")
    # pickle.dump(representations, f)
    # f.close()

    # print("Representations stored in ",db_path,"/",file_name," file. Please delete this file when you add new identities in your database.")

    #----------------------------
    #now, we got representations for facial database

    # if model_name != 'Ensemble':
    df = pd.DataFrame(representations,
                      columns=["identity",
                               "%s_representation" % ('Facenet')])
    # else:  #ensemble learning

    #     columns = ['identity']
    #     [columns.append('%s_representation' % i) for i in model_names]

    #     df = pd.DataFrame(representations, columns=columns)

    df_base = df.copy(
    )  #df will be filtered in each img. we will restore it for the next item.

    resp_obj = []

    # global_pbar = tqdm(range(0,len(img_paths)), desc='Analyzing')
    # for img_path in img_paths:
    # img_path = img_paths[j]

    #find representation for passed image

    #--------------------------------
    #decide input shape
    # input_shape = functions.find_input_shape(custom_model)

    #--------------------------------

    # img = functions.preprocess_face(img = img_path, target_size = input_shape
    #     , enforce_detection = enforce_detection
    #     , detector_backend = detector_backend)

    target_representation = model.predict(image_face)[0, :]

    # print("step 2")
    for k in metric_names:
        distances = []
        for source_representation in representations:

            source_representation = source_representation[1]
            if k == 'cosine':
                distance = dst.findCosineDistance(source_representation,
                                                  target_representation)
            elif k == 'euclidean':
                distance = dst.findEuclideanDistance(source_representation,
                                                     target_representation)
            elif k == 'euclidean_l2':
                distance = dst.findEuclideanDistance(
                    dst.l2_normalize(source_representation),
                    dst.l2_normalize(target_representation))

            distances.append(distance)

        j = 'Facenet'
        df["%s_%s" % (j, k)] = distances

        threshold = dst.findThreshold(j, k)
        df = df.drop(columns=["%s_representation" % (j)])
        df = df[df["%s_%s" % (j, k)] <= threshold]

        df = df.sort_values(by=["%s_%s" % (j, k)],
                            ascending=True).reset_index(drop=True)

        resp_obj.append(df)
        df = df_base.copy()  #restore df for the next iteration
        #---------------------------
    # print("step 3")
    #----------------------------------
    if len(resp_obj) == 1: return resp_obj[0]
    return resp_obj


def find_face_similarity_test(video_path, image_path, dateset_path):

    model = DeepFace.build_model('Facenet')

    vs = FileVideoStream(video_path).start()
    fileStream = True

    # time.sleep(1.0)

    liveness = False

    # frame_rate = np.floor(vs.stream.get(5))

    extract_face = ExtractFace(output_dir=dateset_path, skip=4, model=model)
    liveness_detector = LivenessDetector()

    blurry_count = 0
    i = 0

    image = cv2.imread(image_path)
    image_face = extract_face.detect_face(image)
    target_representation = model.predict(image_face)[0, :]

    distances = []
    # loop over frames from the video stream
    while True:
        # if this is a file video stream, then we need to check if
        # there any more frames left in the buffer to process
        if fileStream and not vs.more():
            break

        # grab the frame from the threaded video file stream
        frame = vs.read()

        if not isinstance(frame, type(None)):
            frame = imutils.resize(frame, width=360)

            # extract_face.extract_by_frame(frame=frame,is_store=False)

            # convert the frame to grayscale and detect blur in it
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # (mean, blurry) = detect_blur_fft(gray,
            #                                  size=60,
            #                                  thresh=10,
            #                                  vis=False)

            # print(f"blurry: {blurry}, mean: {mean}")
            # if not blurry:
            face = extract_face.extract_face_by_frame(frame=frame,
                                                      is_store=False)

            # model
            # print(f"face : {face}")
            if not isinstance(face, type(None)):
                source_representation = model.predict(face)[0, :]
                distance = dst.findEuclideanDistance(source_representation,
                                                     target_representation)
                threshold = dst.findThreshold('Facenet', 'euclidean')

                if distance <= threshold:
                    distances.append(distance)

            if not liveness:
                liveness = liveness_detector.detect_eyeblink_by_frame(
                    frame=gray, is_frame_gray=True)
            else:
                if distance:
                    distances.sort()
                    break
            # else:
            #     blurry_count += 1
            i += 1
        else:
            break
        # print(f"frame {frame}")
    # do a bit of cleanup
    vs.stop()
    print(f"liveness : {liveness}")
    print(f"iter : {i}")
    print(f"blurry_count : {blurry_count}")
    result = {}

    if liveness and distances:
        # print(distances)
        result = {
            "euclidean_distance": float(distances[0]),
            "euclidean_distance_avg": float(np.mean(distances)),
            "similarity": (20 - float(distances[0])) / 20,
            "similarity_avg": (20 - float(np.mean(distances))) / 20,
        }
        # print([type(v) for k, v in result.items()])
        print(result)

    extract_face.clean()
    liveness_detector.clean()
    return result, (i * 0.9) < blurry_count


def test(video_path,
         output_dir,
         model=None,
         skip=4,
         is_store=False,
         only_face=True):
    vs = FileVideoStream(video_path).start()
    fileStream = True
    # time.sleep(1.0)

    extract_face = ExtractFace(output_dir=output_dir, skip=skip, model=model)
    i = 0
    # loop over frames from the video stream
    while True:
        # if this is a file video stream, then we need to check if
        # there any more frames left in the buffer to process
        if fileStream and not vs.more():
            break

        # grab the frame from the threaded video file stream
        frame = vs.read()

        if not isinstance(frame, type(None)):
            frame = imutils.resize(frame, width=360)
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            extract_face.extract_by_frame(frame=frame, is_store=is_store)
            i += 1
        else:
            break
    # do a bit of cleanup
    vs.stop()
    print(f"iter : {i}")