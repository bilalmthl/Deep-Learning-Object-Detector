#!/bin/bash

# 'apples' can be changed to whichever object is being captured
BASE_DIR=~/dataset/apples
CATEGORY=$1

if [ -z "$CATEGORY" ]; then
  echo "Please provide a category name (e.g. desk, hand, bowl)"
  exit 1
fi

# Kill leftover camera processes
sudo pkill -9 raspivid raspistill libcamera-vid libcamera-still 2>/dev/null

# Make and cd into the folder 
mkdir -p $BASE_DIR/$CATEGORY
cd $BASE_DIR/$CATEGORY 

echo "Capturing images in category: $CATEGORY"
echo "Press Enter to capture, Ctrl+C to quit."

# Simple raspistill with preview + keypress
raspistill -k -t 0 -w 224 -h 224 -q 100 \
  -o ${CATEGORY}_%04d.jpg \
  -p 100,100,500,500
