FROM tensorflow/tensorflow:1.15.0-gpu-py3

WORKDIR /opt/stylegan2

RUN pip install torchvision torch requests

ENTRYPOINT ["python", "convert_weight.py"]
