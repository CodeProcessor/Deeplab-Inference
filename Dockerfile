FROM tensorflow/tensorflow:2.5.1-gpu

# Install all the requirements
COPY requirements.txt /tmp
RUN cd /tmp && pip3 install -r requirements.txt

# Copy the project
COPY src /home

WORKDIR /home


