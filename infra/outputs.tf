output "public_ip" {
  value = aws_instance.app.public_ip
}

output "api_url" {
  value = "http://${aws_instance.app.public_ip}:8000"
}

output "artifact_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}