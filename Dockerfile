FROM tensorflow/tensorflow:2.5.1-gpu

# Install all the requirements
COPY requirements.txt /tmp
RUN cd /tmp && pip3 install -r requirements.txt

# Copy the project
COPY src/ /home/src
COPY run.sh /home
COPY server.py /home

WORKDIR /home


