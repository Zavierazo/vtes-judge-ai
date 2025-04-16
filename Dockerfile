FROM python:3.12-slim-bullseye@sha256:87128420453f1b60615c67120fad73dcdf9bed02a26c7c027079bc7a2589cf97


WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]