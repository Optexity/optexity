# Base image
FROM python:3.11-slim

LABEL org.opencontainers.image.source=https://github.com/Optexity/optexity

ARG OPTEXITY_VERSION=dev

ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:1 \
    USER=root \
    PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=0 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# System packages
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y git wget curl nano htop zsh tmux && \
    apt-get install -y xfce4 xfce4-goodies tightvncserver xorg dbus-x11 x11-xserver-utils && \
    apt-get install -y x11-utils x11-apps && \
    apt-get install -y wget curl unzip fonts-liberation libappindicator3-1 libasound2 libnspr4 libnss3 libxss1 libxtst6 xdg-utils && \
    apt-get install -y --no-install-recommends fonts-noto-color-emoji fonts-liberation && \
    apt-get install -y pulseaudio pavucontrol && \
    apt-get install -y udev v4l2loopback-dkms v4l-utils && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /var/tmp/* /tmp/* && \
    apt-get install -y git openssh-client && \
    apt-get install -y \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libdbus-1-3 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
    libpango-1.0-0 libcairo2 libasound2 libatspi2.0-0 libxshmfence1 && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get install -y ca-certificates && update-ca-certificates && \
    apt-get autoremove -y

# Install Python packages
RUN pip install --no-cache-dir playwright patchright

# Install Chromium and Chrome in separate layers
RUN playwright install --with-deps chromium && playwright install --with-deps chrome
RUN patchright install chromium chrome

# VNC server (password: optexity)
RUN mkdir -p /root/.vnc \
    && echo "optexity" | vncpasswd -f > /root/.vnc/passwd \
    && chmod 600 /root/.vnc/passwd \
    && echo '#!/bin/bash\nxrdb $HOME/.Xresources\nstartxfce4 &' > /root/.vnc/xstartup \
    && chmod +x /root/.vnc/xstartup

EXPOSE 5901 9000

# Baseline install — cached, provides all heavy dependencies.
RUN git clone https://github.com/Optexity/optexity.git && \
    cd optexity && git checkout main && \
    pip install -e .

# Always pull latest code and upgrade changed packages
ARG CACHE_BREAK
RUN cd optexity && git pull && git checkout main && git pull && \
    pip install --upgrade -e . && \
    pip cache purge && \
    rm -rf /optexity/.git

ENTRYPOINT ["sh", "-c", "vncserver :1 -geometry 1920x1080 -depth 24 && \
      python /optexity/optexity/inference/child_process.py --child_process_id 0 --port 9000 --is_aws"]
