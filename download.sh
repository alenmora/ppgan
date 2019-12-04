#!/bin/bash

function download_gd() {
  fileid=$1
  filename=$2
  curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
  CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
  curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${fileid}" -o ${filename}
}

fileid="1CrpVlVx12t4yU5q2WXe8lQPtTKjbLzez"
filename=anime_face.zip
download_gd $fileid $filename

mkdir -p ./data
mv $filename ./data/
cd ./data
unzip $filename && rm -rf $filename

#fileid=“PATHTOMODEL”
#filename=“MODELNAME”
#download_gd $fileid $filename

