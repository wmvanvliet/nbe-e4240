FROM quay.io/jupyter/datascience-notebook:notebook-7.1.3
LABEL org.opencontainers.image.authors="Marijn van Vliet <marijn.vanvlie@aalto.fi>"

#some first setup steps need to be run as root user
USER root

# set home environment variable to point to user directory
ENV HOME=/home/$NB_USER

RUN echo "ssh-client and less from apt" \
    && apt-get update \
	&& apt-get install -y ssh-client less \
	&& apt-get clean

RUN echo "g++ from apt" \
	&& apt-get update \
	&& apt-get install -y g++ \
	&& apt-get clean

# disable announcement extension to get rid of newsletter subscription pop-up
RUN jupyter labextension disable "@jupyterlab/apputils-extension:announcements"

# Switch back to jovyan user
USER $NB_USER

# Installing the needed python packages
RUN pip install matplotlib numpy tqdm ipywidgets
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install https://github.com/wmvanvliet/torch-pcdim/archive/refs/heads/main.zip
RUN pip install pymc
RUN pip install hssm

# Clone the repository. First fetch the hash of the latest commit, which will
# invalidate docker's cache when new things are pushed to the repository. See:
# https://stackoverflow.com/questions/36996046
ADD https://api.github.com/repos/wmvanvliet/nbe-e4240/git/refs/heads/main version.json
RUN git init . && \
    git remote add origin https://github.com/wmvanvliet/nbe-e4240.git && \
    git pull origin main
