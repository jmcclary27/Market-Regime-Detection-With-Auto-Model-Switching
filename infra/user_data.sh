#!/bin/bash
apt-get update -y
apt-get install -y docker.io docker-compose-plugin git
systemctl enable docker
systemctl start docker
usermod -aG docker ubuntu