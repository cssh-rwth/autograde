FROM docker.io/python:3.9-bullseye

# prepare working environment
WORKDIR /autograde
RUN mkdir mounts

# install dependencies
RUN apt-get -y update && \
    apt-get -y upgrade

{% if source_dir -%}
# install autograde from source code
COPY {{ source_dir }} src
RUN pip install pip poetry --upgrade &&  \
    cd ./src/autograde &&  \
    poetry config virtualenvs.create false &&  \
    poetry install && \
    cd /autograde
{% else %}
# install autograde from PyPI
RUN pip install pip jupyter-autograde --upgrade
{% endif -%}

{% if requirements -%}
# install custom requirements
ADD {{ requirements|safe }} /tmp/requirements.txt
RUN python3 -m pip install --no-warn-script-location -r /tmp/requirements.txt
{% endif %}

# run autograde
ENTRYPOINT ["python", "-m", "autograde"]