FROM python:3.10-slim

MAINTAINER Siddharth Shah "ss16912@nyu.edu"

WORKDIR /app
ADD . /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txts

EXPOSE 7860

CMD ["python3", "main.py"]


