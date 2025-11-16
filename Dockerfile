FROM python:3.11-slim
WORKDIR /app
RUN apt update && apt install -y git && apt clean
RUN pip install --upgrade pip
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
CMD ["python", "application.py"]
