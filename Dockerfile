FROM continuumio/anaconda3:latest

# install requirements
RUN apt-get --assume-yes update && apt-get --assume-yes upgrade

# set up required directories
WORKDIR /autograde
RUN mkdir src && mkdir target && mkdir context

# load src
COPY . src
RUN pip install -e src[develop]

# create dummy files (will be overwritten by mounted test and notebook)
RUN touch notebook.ipynb && cp src/dummy.py test.py

# run test
ENTRYPOINT ["python", "test.py", "notebook.ipynb", "-c", "./context", "-t", "./target"]