# syntax = docker/dockerfile:experimental@sha256:3c244c0c6fc9d6aa3ddb73af4264b3a23597523ac553294218c13735a2c6cf79
ARG UBUNTU_VERSION=20.04

FROM ubuntu:${UBUNTU_VERSION} as base

# Let us install tzdata painlessly
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install base packages (like Python and pip)
RUN apt update && apt install -y curl zip git lsb-release software-properties-common apt-transport-https vim wget python3 python3-pip
RUN python3 -m pip install --upgrade pip==20.3.4

# Copy Python requirements in and install them
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

# Copy the rest of your training scripts in
COPY . ./

# And tell us where to run the pipeline
ENTRYPOINT ["python3", "-u", "train.py"] 
