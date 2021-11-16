FROM python:3.6.15-slim-buster as base

EXPOSE 5000

RUN pip install --upgrade pip
RUN pip install numpy Flask jsonschema faiss-cpu GPUtil

COPY src /opt/faiss-instant/src

ENTRYPOINT ["python", "/opt/faiss-instant/src/app.py"]
