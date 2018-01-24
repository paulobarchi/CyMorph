#!/bin/sh

echo "Pre-compile script to create directories and download sExtractor exe file"

mkdir imgs
mkdir stamps
mkdir Field
mkdir sExtractor 

cd sExtractor

wget https://www.astromatic.net/download/sextractor/sextractor-2.19.5-1.x86_64.rpm

rpm2cpio sextractor-2.19.5-1.x86_64.rpm | cpio -idmv

cp usr/bin/sex ../

cp ../sex ../maskMaker/
