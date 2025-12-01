FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y python3-pip git && apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000

ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# Make start script executable and provide a lightweight healthcheck
RUN chmod +x start.sh

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
	CMD curl -f http://127.0.0.1:${PORT}/health || exit 1

CMD ["bash", "start.sh"]
