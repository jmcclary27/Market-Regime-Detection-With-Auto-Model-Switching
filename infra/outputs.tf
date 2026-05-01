output "public_ip" {
  value = aws_instance.app.public_ip
}

output "api_url" {
  value = "http://${aws_instance.app.public_ip}:8000"
}