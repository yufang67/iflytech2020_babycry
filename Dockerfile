FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python3.7  python3-pip sudo

RUN useradd -m yufang

RUN chown -R yufang:yufang /home/yufang/

COPY --chown=yufang . /home/yufang/app/

USER yufang

RUN cd /home/yufang/app/ && pip3 install --upgrade pip && pip install -r requirements.txt

WORKDIR /home/yufang/app/
