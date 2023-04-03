# syntax=docker/dockerfile:1
FROM python:3.9

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8000
CMD python ./main.py

# # final configuration
# ENV FLASK_APP=hello
# EXPOSE 8000
# CMD flask run --host 0.0.0.0 --port 8000