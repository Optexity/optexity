#!/usr/bin/env bash
set -x

brew install colima docker docker-buildx gh
mkdir -p ~/.docker/cli-plugins
ln -sfn $(brew --prefix)/opt/docker-buildx/bin/docker-buildx ~/.docker/cli-plugins/docker-buildx