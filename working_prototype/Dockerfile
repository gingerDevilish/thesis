FROM continuumio/anaconda3
MAINTAINER m.kostin@innopolis.ru
 
RUN apt-get update && apt-get install nano -y
RUN git clone https://github.com/kostin-001/tmp.git
RUN mv tmp/material.tar.gz ..
RUN tar -xzf material.tar.gz
RUN rm -rf material.tar.gz
RUN pip install tensorflow && pip install opencv-contrib-python && pip install keras

EXPOSE 22 80

CMD ["/bin/bash"]