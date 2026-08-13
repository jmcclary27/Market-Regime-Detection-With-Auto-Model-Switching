output "inference_bucket_name" {
  description = "Dedicated versioned S3 bucket for immutable requests, inputs, bundles, and outputs."
  value       = aws_s3_bucket.inference.bucket
}

output "ecr_repository_url" {
  description = "Push the dedicated Lambda image here before setting deploy_lambda=true."
  value       = aws_ecr_repository.inference.repository_url
}

output "inference_lambda_name" {
  description = "Null until deploy_lambda=true."
  value       = try(aws_lambda_function.inference[0].function_name, null)
}

output "live_sim_lambda_name" {
  description = "Null until deploy_lambda=true."
  value       = try(aws_lambda_function.live_sim[0].function_name, null)
}

output "inference_dlq_url" {
  description = "SQS destination for malformed or failed asynchronous Lambda requests."
  value       = aws_sqs_queue.inference_dlq.url
}

output "experiment_dashboard_url" {
  description = "Public read-only dashboard URL when the frozen experiment is enabled."
  value       = try("https://${aws_cloudfront_distribution.dashboard[0].domain_name}", null)
}
