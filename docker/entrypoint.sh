#!/usr/bin/env bash
set -e

if [ "${APP_ENV:-prod}" = "dev" ]; then
  exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.dev.conf
else
  exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.prod.conf
fi