FROM docker.io/python:3.9-bullseye

# install dependencies
RUN apt-get -y update && \
    apt-get -y upgrade
RUN pip install pip jupyter-autograde --upgrade

# prepare working environment
WORKDIR /autograde
RUN mkdir mounts

{% if requirements -%}
# install custom requirements
ADD {{ requirements|safe }} /tmp/requirements.txt
RUN python3 -m pip install --no-warn-script-location -r /tmp/requirements.txt
{% endif %}

# run autograde
ENTRYPOINT ["python", "-m", "autograde"]