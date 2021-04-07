FROM tensorflow/tensorflow

COPY . /workspace
WORKDIR /workspace

RUN apt-get update -y
RUN apt-get install -y build-essential gcc cmake libprotobuf-dev protobuf-compiler libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg

RUN pip install --upgrade pip

RUN pip install -r requirements.txt