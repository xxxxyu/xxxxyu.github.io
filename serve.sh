#!/usr/bin/env bash
set -e
exec zola serve --extra-watch-path data "$@"
