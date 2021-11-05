#!/usr/bin/env bash

sudo apt-get remove docker docker-engine docker.io containerd runc

sudo apt-get update

sudo apt-get install apt-transport-https ca-certificates curl gnupg

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io

apt-cache madison docker-ce

sudo apt-get install docker-ce=5:19.03.12~3-0~ubuntu-bionic docker-ce-cli=5:19.03.12~3-0~ubuntu-bionic containerd.io

sudo docker run hello-world


#INSTALL_DIR=~
#source launch_container.sh ${INSTALL_DIR}/txt_db ${INSTALL_DIR}/img_db \
#    ${INSTALL_DIR}/finetune ${INSTALL_DIR}/pretrained
