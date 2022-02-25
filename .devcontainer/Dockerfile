# syntax=docker/dockerfile:1.2
ARG     CONDA_VERSION
FROM    continuumio/anaconda3:${CONDA_VERSION}

# Create a non-root user
ARG     USER
ARG     USERNAME=${USER}
ARG     GROUPNAME=${USERNAME}
ARG     USER_UID=1000
ARG     USER_GID=1000
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
RUN     set -x && \
        sudo --preserve-env apt update && \
        sudo --preserve-env apt install -y \
                vim \
                less \
                git \
                wget \
                tar \
                && \
        sudo --preserve-env apt clean && \
        sudo rm -fr /var/lib/apt/lists/* && \
        :
# Append aliases to .bashrc
RUN     set -x && \
                echo "alias l='ls'" >> ~/.bashrc
RUN     set -x && \
                echo "alias ll='ls -l'" >> ~/.bashrc
RUN     set -x && \
                echo "alias lll='ls -lA'" >> ~/.bashrc
# Install fuzzy finder
RUN     git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
RUN     cd ~/.fzf && ./install

# Install exercism
#  First change to bash shell to be albot to echo multiline content
#  into a file
SHELL ["/bin/bash", "-c"]
RUN     set -x && \
        wget \
                https://github.com/exercism/cli/releases/download/v3.0.13/exercism-3.0.13-linux-x86_64.tar.gz \
                -O ~/exercism-3.0.13-linux-x86_64.tar.gz && \
        mkdir -p ~/bin && \
        mv ~/exercism-3.0.13-linux-x86_64.tar.gz ~/bin/ && \
        tar xf ~/bin/exercism-3.0.13-linux-x86_64.tar.gz -C ~/bin/ && \
        echo $'\n\
        if [ -f ~/.config/exercism/exercism_completion.bash ]; then \n\
                source ~/.config/exercism/exercism_completion.bash \n\
        fi' >> ~/.bashrc && \
        echo 'export PATH=~/bin:$PATH' >> ~/.bashrc && \
        :
