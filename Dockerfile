FROM magnetcoop/mxnet-clj-gpu

RUN set -ex; \
    apt-get update && \
    apt-get install -y --no-install-recommends x11-apps  && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/magnet/app

ENTRYPOINT ["/usr/local/bin/run-as-user.sh"]