FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python3.7  python3-pip sudo

RUN apt-get install -y libsndfile1

RUN apt-get install -y libgl1-mesa-dev

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

RUN useradd -m yufang

RUN chown -R yufang:yufang /home/yufang/

COPY --chown=yufang . /home/yufang/app/

USER yufang

RUN cd /home/yufang/app/ && pip3 install --upgrade pip

ENV PATH="/home/yufang/.local/bin":${PATH}

RUN cd /home/yufang/app/ && pip3 install -r requirements.txt

WORKDIR /home/yufang/app/
