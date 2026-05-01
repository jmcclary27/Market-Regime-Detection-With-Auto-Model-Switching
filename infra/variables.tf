variable "aws_region" {
  default = "us-east-1"
}

variable "public_key_path" {
  default = "~/.ssh/id_rsa.pub"
}

variable "my_ip_cidr" {
  description = "Your public IP in CIDR format"
}