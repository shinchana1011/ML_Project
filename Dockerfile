FROM python:3.11-slim-bullseye
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt
COPY . /app
CMD ["python", "application.py"]
