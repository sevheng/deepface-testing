import os
import time

import cv2
import dlib
import imutils
import numpy as np
from deepface import DeepFace
from imutils import face_utils
from imutils.video import FileVideoStream, VideoStream
from scipy.spatial import distance as dist


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

    def detect_eyeblink_by_frame(self, frame):
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = self.detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = self.predictor(gray, rect)
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

    protoPath = ".deepface/weights/deploy.prototxt"
    modelPath = ".deepface/weights/res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # open a pointer to the video file stream and initialize the total
    # number of frames read and saved thus far
    read = 0
    saved = 0

    # of frames to skip before applying face detection
    skip = 4

    # minimum probability to filter weak detections
    confidence = 0.5

    output_dir = "dataset"

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def clean(self):
        self.read = 0
        self.saved = 0
        for f in os.listdir(self.output_dir):
            os.remove(os.path.join(self.output_dir, f))

    def extract_face_by_frame(self, frame):
        self.read += 1

        # check to see if we should process this frame
        if self.read % self.skip != 0:
            return

        # grab the frame dimensions and construct a blob from the frame
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (160, 160)), 1.0,
                                     (160, 160), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        detections = self.net.forward()

        # ensure at least one face was found
        if len(detections) > 0:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            # ensure that the detection with the largest probability also
            # means our minimum probability test (thus helping filter out
            # weak detections)
            if confidence > self.confidence:
                # compute the (x, y)-coordinates of the bounding box for
                # the face and extract the face ROI
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = frame[startY:endY, startX:endX]

                # write the frame to disk
                p = os.path.sep.join(
                    [self.output_dir, "{}.jpg".format(self.saved)])
                cv2.imwrite(p, face)
                self.saved += 1
                print("[INFO] saved {} to disk".format(p))

    @classmethod
    def extract(cls, video_path, output_dir):
        vs = FileVideoStream(video_path).start()
        fileStream = True
        # time.sleep(1.0)

        extract_face = cls(output_dir=output_dir)

        # loop over frames from the video stream
        while True:
            # if this is a file video stream, then we need to check if
            # there any more frames left in the buffer to process
            if fileStream and not vs.more():
                break

            # grab the frame from the threaded video file stream
            frame = vs.read()

            if not isinstance(frame, type(None)):
                extract_face.extract_face_by_frame(frame=frame)

        # do a bit of cleanup
        vs.stop()


def find_face_similarity(video_path, image_path, dateset_path):

    vs = FileVideoStream(video_path).start()
    fileStream = True

    # time.sleep(1.0)

    liveness = False

    extract_face = ExtractFace(output_dir=dateset_path)
    liveness_detector = LivenessDetector()

    # loop over frames from the video stream
    while True:
        # if this is a file video stream, then we need to check if
        # there any more frames left in the buffer to process
        if fileStream and not vs.more():
            break

        # grab the frame from the threaded video file stream
        frame = vs.read()

        if not isinstance(frame, type(None)):
            extract_face.extract_face_by_frame(frame=frame)
            if not liveness:
                liveness = liveness_detector.detect_eyeblink_by_frame(
                    frame=frame)

        # print(f"frame {frame}")

    # do a bit of cleanup
    vs.stop()
    print(f"liveness : {liveness}")
    result = {}
    if liveness:
        df = DeepFace.find(image_path, db_path=dateset_path, model_name='Facenet',
                           distance_metric='euclidean', detector_backend='ssd', enforce_detection=False)

        if len(df) > 20:
            df = df.iloc[:20]
            result = {
                "similarity": (10 - df.iloc[0]['Facenet_euclidean']) / 10,
                "avg_similarity": (10 - df['Facenet_euclidean'].mean()) / 10
            }
            print(df)
            print(result)

    extract_face.clean()
    liveness_detector.clean()
    return result
