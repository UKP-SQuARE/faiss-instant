FROM python:3.6.15-slim-buster as base

EXPOSE 7000

RUN pip install --upgrade pip
RUN pip install numpy Flask jsonschema faiss-cpu

COPY resources /opt/faiss-instant/resources
COPY src /opt/faiss-instant/src

ENTRYPOINT ["python", "/opt/faiss-instant/src/app.py"]
