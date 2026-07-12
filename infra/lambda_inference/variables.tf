variable "aws_region" {
  description = "AWS region for the dedicated inference stack."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Short lowercase project identifier used in AWS resource names."
  type        = string
  default     = "market-regime"
}

variable "environment" {
  description = "Environment identifier used in AWS resource names and tags."
  type        = string
  default     = "dev"
}

variable "deploy_lambda" {
  description = "Set true only after an image has been pushed to the dedicated ECR repository."
  type        = bool
  default     = false
}

variable "lambda_image_uri" {
  description = "Immutable ECR image URI (prefer @sha256 digest) used only when deploy_lambda=true."
  type        = string
  default     = ""

  validation {
    condition     = !var.deploy_lambda || trimspace(var.lambda_image_uri) != ""
    error_message = "lambda_image_uri is required when deploy_lambda is true."
  }
}

variable "lambda_architecture" {
  description = "Lambda image architecture. arm64 is cheaper; use x86_64 if a native ML wheel requires it."
  type        = string
  default     = "arm64"

  validation {
    condition     = contains(["arm64", "x86_64"], var.lambda_architecture)
    error_message = "lambda_architecture must be arm64 or x86_64."
  }
}

variable "lambda_memory_mb" {
  description = "Memory allocation for the ML inference runtime."
  type        = number
  default     = 2048
}

variable "live_sim_memory_mb" {
  description = "Memory allocation for the serial paper live-simulation executor."
  type        = number
  default     = 1024
}

variable "lambda_timeout_seconds" {
  description = "Maximum single inference duration; Lambda permits at most 900 seconds."
  type        = number
  default     = 300

  validation {
    condition     = var.lambda_timeout_seconds >= 1 && var.lambda_timeout_seconds <= 900
    error_message = "lambda_timeout_seconds must be between 1 and 900."
  }
}

variable "live_sim_timeout_seconds" {
  description = "Maximum single live-simulation event duration."
  type        = number
  default     = 120

  validation {
    condition     = var.live_sim_timeout_seconds >= 1 && var.live_sim_timeout_seconds <= 900
    error_message = "live_sim_timeout_seconds must be between 1 and 900."
  }
}

variable "lambda_ephemeral_storage_mb" {
  description = "Temporary storage for the downloaded parquet input and model bundle."
  type        = number
  default     = 512

  validation {
    condition     = var.lambda_ephemeral_storage_mb >= 512 && var.lambda_ephemeral_storage_mb <= 10240
    error_message = "lambda_ephemeral_storage_mb must be between 512 and 10240."
  }
}

variable "run_artifact_retention_days" {
  description = "Retention for versioned run inputs and output artifacts."
  type        = number
  default     = 90
}

variable "request_retention_days" {
  description = "Retention for request manifests."
  type        = number
  default     = 90
}

variable "model_bundle_retention_days" {
  description = "Retention for model bundles referenced by immutable inference runs."
  type        = number
  default     = 180
}

variable "noncurrent_version_retention_days" {
  description = "Retention for noncurrent S3 object versions created by accidental retries/overwrites."
  type        = number
  default     = 30
}

variable "log_retention_days" {
  description = "CloudWatch Logs retention to constrain observability cost."
  type        = number
  default     = 14
}

variable "dlq_retention_seconds" {
  description = "How long failed S3 invocation records remain in the SQS DLQ."
  type        = number
  default     = 1209600
}
