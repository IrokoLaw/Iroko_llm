FROM python:3.12

RUN apt-get update && \
    apt-get install -y wget curl && \
    apt-get clean &&\
    apt-get remove --purge -y 

WORKDIR /app

COPY ./requirements.txt /tmp/requirements.txt

RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt &&\
    rm -r /tmp/requirements.txt

COPY ./src src
COPY ./.env .env
COPY ./config.py config.py
COPY ./fields.py fields.py
COPY ./main.py main.py
COPY ./fields.py fields.py
RUN mkdir data

EXPOSE 8000
CMD ["--host", "0.0.0.0", "--port", "8000"]
ENTRYPOINT [ "uvicorn", "main:app" ]