#!/bin/bash

source env.sh
python3 update_website.py && \
  git stage index.html && \
  git stage outputs/daily/* && \
  git commit -m "daily update" && \
  git push
