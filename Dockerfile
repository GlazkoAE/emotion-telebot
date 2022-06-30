FROM python:3.8

ARG APP_DIR=/app
ENV key secret
WORKDIR /app
COPY requirements.txt requirements.txt

RUN apt-get update -y
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libfontconfig1
RUN apt-get install -y libxext6 libgl1-mesa-glx
RUN apt-get install libsndfile1

RUN pip install -U pip
RUN pip install -U setuptools
RUN pip install -r requirements.txt

COPY . .

CMD python3 -m run_bot -key ${key}