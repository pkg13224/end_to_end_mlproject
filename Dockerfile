# I'm basically saying that take
# 2:18
# a base image of python 3.8 slim Buster so whatever happen is that from the docker Hub it will bring up this python
# 2:25
# 3.8 version of Linux machine I'll not say Linux machine Linux base image so
# 2:32
# that it can be taken and can be done with respect to the deployment it will not take machine okay it will be taking
# 2:38
# a base image of a Linux environment and uh we are creating a working directory
# 2:43
# called as app okay then uh the next step is basically copying this entire project
# 2:49
# into this app folder okay the entire project into this app folder over here and then the next step is a run command
# 2:56
# where we are updating all the packages after probably uh doing before doing the
# 3:02
# deployment in that specific server it can be a Linux machine okay and we will definitely use a Ubuntu machine over
# 3:08
# there and then we will go ahead and do the installment of the entire requirement.txt so once that install
# 3:14
# emission takes place this is the command that is used to run the file that is app.py python3 app.py just specifying
# 3:22
# that we are working in Python 3 version 3.8 version and then the app.py is basically my file name which it will be
# 3:29
# running it is the same project that we have actually done student performance indicator right that same project see



FROM python:3.9-slim
WORKDIR /app
COPY . /app

EXPOSE 8080
RUN pip install -r requirements.txt
CMD ["python3", "app.py"]


 