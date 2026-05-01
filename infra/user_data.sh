#!/bin/bash
exec > /var/log/user-data.log 2>&1

apt-get update -y
apt-get install -y docker.io docker-compose git

systemctl start docker
systemctl enable docker

usermod -aG docker ubuntu