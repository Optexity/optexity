FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
ENV XAUTHORITY=/home/optexity/.Xauthority
# Software rendering for WebGL on GPU-less EC2 instances
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV GALLIUM_DRIVER=llvmpipe
ENV MESA_GL_VERSION_OVERRIDE=4.5
ENV MESA_GLSL_VERSION_OVERRIDE=450
ENV LIBGL_DRI3_DISABLE=1
ENV SCREEN_WIDTH=1440
ENV SCREEN_HEIGHT=900

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo git wget curl nano htop zsh tmux \
    xvfb openbox x11-utils \
    freerdp2-x11 \
    x11vnc novnc websockify \
    supervisor wmctrl \
    python3 python3-pip \
    xclip wl-clipboard \
    && pip3 install --break-system-packages fastapi uvicorn \
    && rm -rf /var/lib/apt/lists/*

# System packages (installed as root)
RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip fonts-liberation libappindicator3-1 libasound2 libnspr4 libnss3 libxss1 libxtst6 xdg-utils \
    fonts-noto-color-emoji fonts-liberation \
    pulseaudio pavucontrol \
    python3-tk python3-dev \
    udev v4l2loopback-dkms v4l-utils \
    libgl1-mesa-dri libgl1 libegl-mesa0 libgles2 \
    mesa-utils mesa-vulkan-drivers libvulkan1 \
    libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libdbus-1-3 libgtk-3-0 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
    libpango-1.0-0 libcairo2 libasound2 libatspi2.0-0 libxshmfence1 ca-certificates \
    && update-ca-certificates && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /var/tmp/* /tmp/*

# Google Chrome (amd64 only, skip on arm64) ─────────────────────
RUN if [ "$(dpkg --print-architecture)" = "amd64" ]; then \
        wget -qO /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
        && apt-get install -y /tmp/chrome.deb \
        && rm /tmp/chrome.deb; \
    fi && apt-get update && apt-get install -y chromium && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages --no-cache-dir playwright patchright

# Install Chromium and Chrome in separate layers
RUN playwright install --with-deps chromium
RUN patchright install chromium

# Create non-root user 'optexity' with passwordless sudo
RUN groupadd -r optexity && \
    useradd -r -g optexity -m -d /home/optexity -s /bin/bash optexity && \
    echo "optexity ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/optexity && \
    chmod 0440 /etc/sudoers.d/optexity && \
    touch /home/optexity/.Xauthority

# Make browser directories accessible to the non-root user
RUN chmod -R o+rx /ms-playwright

# Create DRI device directory for Mesa software rendering
RUN mkdir -p /dev/dri && chmod 777 /dev/dri

EXPOSE 8000 8080 9000

WORKDIR /home/optexity

# Baseline install — cached, provides all heavy dependencies.
RUN git clone https://github.com/Optexity/optexity.git && \
    cd optexity && git checkout main && \
    pip install --break-system-packages -e .

# Give the non-root user ownership of the project and writable directories
RUN chown -R optexity:optexity /home/optexity && \
    chmod 1777 /tmp

ARG CACHE_BREAK
RUN echo "CACHE_BREAK: $CACHE_BREAK"

# Copy system config files as root before switching user
COPY openbox-rc.xml.template /etc/xdg/openbox/rc.xml.template
COPY supervisord.dev.conf /etc/supervisor/conf.d/supervisord.dev.conf
COPY supervisord.prod.conf /etc/supervisor/conf.d/supervisord.prod.conf

RUN git config --global --add safe.directory /home/optexity/optexity && \
    cd optexity && git pull && git checkout ui_agents && git pull && \
    pip install --upgrade --break-system-packages -e . && \
    pip cache purge && \
    rm -rf /optexity/.git

ENV APP_ENV=prod
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER optexity
ENV USER=optexity \
    HOME=/home/optexity

ENTRYPOINT ["/entrypoint.sh"]