# syntax=docker/dockerfile:1.2
ARG     MAMBAFORGE_VERSION=4.11.0-4
FROM    condaforge/mambaforge:${MAMBAFORGE_VERSION}

# Create a non-root user
ARG     USER
ARG     USERNAME=${USER}
ARG     GROUPNAME=${USERNAME}
ARG     USER_UID=1010
ARG     USER_GID=1010
RUN     set -x && \
        groupadd \
                --gid ${USER_GID} \
                ${GROUPNAME} \
                && \
        useradd \
                --uid ${USER_UID} \
                --gid ${USER_GID} \
                --create-home \
                ${USERNAME} \
        && \
        :

RUN     set -x && \
        apt update && \
        apt upgrade -y && \
        apt install -y \
                sudo \
        && \
        echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} && \
        chmod 0440 /etc/sudoers.d/${USERNAME} && \
        apt clean && \
        rm -fr /var/lib/apt/lists/* && \
        :
# Switch to the non-root user
USER    ${USERNAME}
# Install stuff...
# build-essentials is installed in case we need to compile some wheel packages
RUN     set -x && \
        sudo --preserve-env apt update && \
        sudo --preserve-env apt install -y \
                vim \
                less \
                git \
                wget \
                tar \
                python-opengl \
                swig3.0 \
                build-essential \
                xvfb \
                && \
        sudo --preserve-env apt clean && \
        sudo rm -fr /var/lib/apt/lists/* && \
        :
# symbolic link: swig3.0 -> swig
RUN     set -x && \
                sudo ln -s /usr/bin/swig3.0 /usr/bin/swig && \
                :
# Append aliases to .bashrc
RUN     set -x && \
                echo "alias l='ls'" >> ~/.bashrc && \
                :
RUN     set -x && \
                echo "alias ll='ls -l'" >> ~/.bashrc && \
                :
RUN     set -x && \
                echo "alias lll='ls -lA'" >> ~/.bashrc && \
                :
# Install fuzzy finder
RUN     git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf && :
RUN     cd ~/.fzf && ./install && :

# install jupyterlab
COPY --chown=${USERNAME}:${GROUPNAME} environment-linux-64.lock .
# In case it hasn't been created yet, create the lock file with the following command:
#       conda lock -p linux-64 -k explicit -f conda_environment.yml --filename-template 'environment-{platform}.lock'
RUN --mount=type=cache,target=/opt/conda/pkgs mamba create --name deepReinforcement --file environment-linux-64.lock && \
        conda clean -afy && \
        :
