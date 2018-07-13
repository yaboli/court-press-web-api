FROM ubuntu:latest
MAINTAINER Yabo Li "liyabo@FOUNDER.COM.CN"
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip
ENV LANG C.UTF-8
COPY /resources /app/resources
COPY app.py /app/app.py
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python3"]
CMD ["app.py"]