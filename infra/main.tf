provider "aws" {
  region = var.aws_region
}

resource "aws_key_pair" "app_key" {
  key_name   = "market-regime-key"
  public_key = file(var.public_key_path)
}

resource "aws_security_group" "app_sg" {
  name = "market-regime-sg"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip_cidr]
  }

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [var.my_ip_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

data "aws_ami" "ubuntu_arm" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-arm64-server-*"]
  }
}

resource "aws_instance" "app" {
  ami                    = data.aws_ami.ubuntu_arm.id
  instance_type          = "t4g.small"
  key_name               = aws_key_pair.app_key.key_name
  vpc_security_group_ids = [aws_security_group.app_sg.id]

  root_block_device {
    volume_size = 25
    volume_type = "gp3"
  }

  user_data = file("${path.module}/user_data.sh")

  tags = {
    Name = "market-regime-app"
  }
}

resource "aws_s3_bucket" "artifacts" {
  bucket = "market-regime-artifacts-${random_id.bucket_suffix.hex}"

  tags = {
    Name = "market-regime-artifacts"
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}