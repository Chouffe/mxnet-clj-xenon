#!/usr/bin/env bash

nvidia-docker run \
	      --name xenon-run \
	      --rm \
	      --volume "$PWD/app:/home/magnet/app" \
	      --volume "$HOME/.m2:/home/magnet/.m2" \
	      --volume "$HOME/.m2:/root/.m2" \
	      --interactive \
	      --tty \
	      magnetcoop/mxnet-clj-gpu \
	      lein run
