FROM python:3.9-buster

# install requirements
RUN apt-get --assume-yes update && apt-get --assume-yes upgrade

# set up required directories
WORKDIR /autograde
RUN mkdir src && mkdir target && mkdir context

# load src & install
ADD ./requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . src
RUN pip install -e src

# create dummy files (will be overwritten by mounted test and notebook)
RUN touch notebook.ipynb && cp src/dummy.py test.py

# run test
ENTRYPOINT ["python", "test.py", "notebook.ipynb", "--context", "./context", "--target", "./target"]