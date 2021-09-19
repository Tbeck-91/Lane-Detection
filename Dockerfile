FROM tensorflow/tensorflow:2.2.0

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    python3-opencv\
    wget

# Add new user to avoid running as root
RUN useradd -ms /bin/bash tensorflow
USER tensorflow
WORKDIR /home/tensorflow

# Copy this version of of the model garden into the image
COPY --chown=tensorflow ./tf2 /home/tensorflow/models

# Compile protobuf configs
RUN (cd /home/tensorflow/models/research/ && protoc object_detection/protos/*.proto --python_out=.)
WORKDIR /home/tensorflow/models/research/

RUN cp object_detection/packages/tf2/setup.py ./
ENV PATH="/home/tensorflow/.local/bin:${PATH}"

RUN python -m pip install -U pip
RUN python -m pip install .

RUN pip install -U tensorflow==2.2.0

# Install additional pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# copy app project files
WORKDIR /home/tensorflow/app
COPY --chown=tensorflow ./application /home/tensorflow/app
COPY --chown=tensorflow ./saved_model /home/tensorflow/app/saved_model

# run app
CMD ["python", "app.py"]
ENV TF_CPP_MIN_LOG_LEVEL 3
