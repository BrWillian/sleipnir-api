FROM tensorflow/tensorflow:latest

RUN apt-get update && \
apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx

WORKDIR /app

COPY . /app

RUN pip3 install --upgrade pip
RUN pip3 --no-cache-dir install -r requeriments.txt

EXPOSE 5000

ENTRYPOINT ["python3"]
CMD ["run.py"]
