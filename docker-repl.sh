#!/usr/bin/env bash

nvidia-docker run \
	      --name xenon-repl \
	      --rm \
	      --env "DISPLAY" \
	      --net host \
	      --volume "$HOME/.Xauthority:/home/magnet/.Xauthority:rw" \
	      --volume "$PWD/app:/home/magnet/app" \
	      --volume "$HOME/.m2:/home/magnet/.m2" \
	      --volume "$HOME/.m2:/root/.m2" \
	      --interactive \
	      --tty \
	      xenon-gpu \
	      lein repl :headless :host 0.0.0.0 :port 3133
