FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y python3-pip git && apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# RunPod Serverless handler entrypoint
ENV RUNPOD_HANDLER=handler

CMD ["python", "-u", "handler.py"]
