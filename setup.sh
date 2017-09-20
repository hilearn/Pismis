#!/bin/sh

add-apt-repository -y ppa:ubuntugis

apt-get update && apt-get install -y \
	gfortran=4:5.3.1-1ubuntu1 \
	libgdal-dev=2.1.3+dfsg-1~xenial2 \
	python3-dev \
	python3-virtualenv
