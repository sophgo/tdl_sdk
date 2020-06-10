#!/bin/bash

# DO NOT SOURCE THIS FILE
cd $(dirname $0)

bmtap_docker="$(sudo docker image ls | grep "gitlab-ai.bitmain.vip:4567/bitmain/bmtap2/ubuntu   1.5.10 " | tee /dev/tty)"

if [[ "$bmtap2_docker" == 0 ]]; then
    echo -e "No bmtap2 docker found! Build docker gitlab-ai.bitmain.vip:4567/bitmain/bmtap2/ubuntu:1.5.10.1 first!"
    exit
fi
CI_REGISTRY_IMAGE=gitlab-ai.bitmain.vip:4567/bitmain/bmtap2/ubuntu:1.5.10.1

docker build -t ${CI_REGISTRY_IMAGE} -f Dockerfile.dev .
docker push ${CI_REGISTRY_IMAGE}
