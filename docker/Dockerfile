# Base image
FROM python:3.11-slim

LABEL org.opencontainers.image.source=https://github.com/Optexity/optexity

ARG OPTEXITY_VERSION=dev

ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:1 \
    PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=0 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    # Software rendering for WebGL on GPU-less EC2 instances
    LIBGL_ALWAYS_SOFTWARE=1 \
    GALLIUM_DRIVER=llvmpipe \
    MESA_GL_VERSION_OVERRIDE=4.5 \
    MESA_GLSL_VERSION_OVERRIDE=450 \
    LIBGL_DRI3_DISABLE=1

# System packages (installed as root)
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y sudo git wget curl nano htop zsh tmux && \
    apt-get install -y xfce4 xfce4-goodies tightvncserver xorg dbus-x11 x11-xserver-utils && \
    apt-get install -y x11-utils x11-apps && \
    apt-get install -y wget curl unzip fonts-liberation libappindicator3-1 libasound2 libnspr4 libnss3 libxss1 libxtst6 xdg-utils && \
    apt-get install -y --no-install-recommends fonts-noto-color-emoji fonts-liberation && \
    apt-get install -y pulseaudio pavucontrol && \
    apt-get install -y udev v4l2loopback-dkms v4l-utils && \
    apt-get install -y libgl1-mesa-dri libgl1 libegl-mesa0 libgles2 \
    mesa-utils mesa-vulkan-drivers libvulkan1 xvfb && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /var/tmp/* /tmp/* && \
    apt-get install -y git openssh-client && \
    apt-get install -y \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libdbus-1-3 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
    libpango-1.0-0 libcairo2 libasound2 libatspi2.0-0 libxshmfence1 && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get install -y ca-certificates && update-ca-certificates && \
    apt-get autoremove -y

RUN apt-get update && apt-get install -y \
    gnupg lsb-release \
 && wget -qO- https://dl.google.com/linux/linux_signing_key.pub \
    | gpg --dearmor -o /usr/share/keyrings/google-linux-signing-key.gpg \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-linux-signing-key.gpg] http://dl.google.com/linux/chrome/deb/ stable main" \
    > /etc/apt/sources.list.d/google-chrome.list \
 && apt-get update \
 && apt-get install -y \
    google-chrome-stable \
    chromium \
    fonts-liberation \
    libnss3 \
    libatk-bridge2.0-0 \
    libxkbcommon0 \
    libgtk-3-0 \
    libgbm1 \
 && rm -rf /var/lib/apt/lists/*

# Create non-root user 'optexity' with passwordless sudo
RUN groupadd -r optexity && \
    useradd -r -g optexity -m -d /home/optexity -s /bin/bash optexity && \
    echo "optexity ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/optexity && \
    chmod 0440 /etc/sudoers.d/optexity

# Install Python packages
RUN pip install --no-cache-dir playwright patchright

# Install Chromium and Chrome in separate layers
RUN playwright install --with-deps chromium && playwright install --with-deps chrome
RUN patchright install chromium chrome

# Make browser directories accessible to the non-root user
RUN chmod -R o+rx /ms-playwright

# Create DRI device directory for Mesa software rendering
RUN mkdir -p /dev/dri && chmod 777 /dev/dri

# VNC server (password: optexity) — configured for the non-root user
RUN mkdir -p /home/optexity/.vnc \
    && echo "optexity" | vncpasswd -f > /home/optexity/.vnc/passwd \
    && chmod 600 /home/optexity/.vnc/passwd \
    && echo '#!/bin/bash\nxrdb $HOME/.Xresources\nstartxfce4 &' > /home/optexity/.vnc/xstartup \
    && chmod +x /home/optexity/.vnc/xstartup \
    && chown -R optexity:optexity /home/optexity/.vnc

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

# Give the non-root user ownership of the project and writable directories
RUN chown -R optexity:optexity /optexity && \
    chown -R optexity:optexity /home/optexity && \
    chmod 1777 /tmp

# Switch to non-root user
USER optexity
ENV USER=optexity \
    HOME=/home/optexity

ENTRYPOINT ["sh", "-c", "vncserver :1 -geometry 1920x1080 -depth 24 && \
      python /optexity/optexity/inference/child_process.py --child_process_id 0 --port 9000 --is_aws"]

mkdir -p "$HOME" && printf "set tabsize 4\nset tabstospaces\n" >> "$HOME/.nanorc"
vncserver :1 -geometry 1920x1080 -depth 24
optexity inference --port 9002 --child_process_id 0


/usr/bin/google-chrome --no-first-run --no-default-browser-check --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-profile --use-gl=swiftshader --use-angle=swiftshader-webgl --enable-webgl --enable-webgl2 --disable-dev-shm-usage --ignore-gpu-blocklist --disable-gpu-sandbox --in-process-gpu --gpu-sandbox-failures-fatal=no --disable-gpu

/usr/bin/google-chrome --no-first-run --no-default-browser-check --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-profile --disable-dev-shm-usage --use-gl=swiftshader --use-angle=swiftshader-webgl --enable-webgl --enable-webgl2 --ignore-gpu-blocklist --disable-gpu-sandbox --in-process-gpu


sudo docker run -it --rm \
  --security-opt seccomp=./seccomp_profile.json \
  --user optexity \
  --platform linux/amd64 \
  -p 5901:5901 \
  -p 9002:9002 \
  -e USE_PLAYWRIGHT_BROWSER="False" \
  -e GOOGLE_API_KEY="AIzaSyABTjn3DGZrjoLe6eMOD6l-bFNfmSrc8EU" \
  -e API_KEY="34328cf9-94d4-4595-8860-4ccd0ad70649" \
  -e DEPLOYMENT=dev \
  --entrypoint bash \
  ghcr.io/optexity/opinference-dev:latest