terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}

locals {
  name_prefix            = "${var.project_name}-${var.environment}"
  lambda_name            = "${local.name_prefix}-inference"
  live_sim_lambda_name   = "${local.name_prefix}-live-sim"
  request_prefix         = "inference/requests/"
  request_suffix         = "/request.json"
  live_sim_output_prefix = "inference/live-sim/runs/"
  completion_suffix      = "/completed.json"
}

resource "aws_s3_bucket" "inference" {
  bucket_prefix = "${local.name_prefix}-inference-"
  force_destroy = false

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "event-driven-inference"
  }
}

resource "aws_s3_bucket_ownership_controls" "inference" {
  bucket = aws_s3_bucket.inference.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_public_access_block" "inference" {
  bucket                  = aws_s3_bucket.inference.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "inference" {
  bucket = aws_s3_bucket.inference.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "inference" {
  bucket = aws_s3_bucket.inference.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "inference" {
  bucket = aws_s3_bucket.inference.id

  rule {
    id     = "expire-run-artifacts"
    status = "Enabled"

    filter {
      prefix = "inference/runs/"
    }

    expiration {
      days = var.run_artifact_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = var.noncurrent_version_retention_days
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  rule {
    id     = "expire-request-manifests"
    status = "Enabled"

    filter {
      prefix = "inference/requests/"
    }

    expiration {
      days = var.request_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = var.noncurrent_version_retention_days
    }
  }

  rule {
    id     = "expire-live-sim-run-artifacts"
    status = "Enabled"

    filter {
      prefix = "inference/live-sim/runs/"
    }

    expiration {
      days = var.run_artifact_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = var.noncurrent_version_retention_days
    }
  }

  rule {
    id     = "expire-old-model-bundles"
    status = "Enabled"

    filter {
      prefix = "inference/model-bundles/"
    }

    expiration {
      days = var.model_bundle_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = var.noncurrent_version_retention_days
    }
  }

  rule {
    id     = "expire-noncurrent-live-sim-state"
    status = "Enabled"

    filter {
      prefix = "live-sim/state/"
    }

    noncurrent_version_expiration {
      noncurrent_days = var.noncurrent_version_retention_days
    }
  }
}

resource "aws_ecr_repository" "inference" {
  name                 = "${local.name_prefix}-inference"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "event-driven-inference"
  }
}

resource "aws_ecr_lifecycle_policy" "inference" {
  repository = aws_ecr_repository.inference.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep the three most recent immutable inference images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 3
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

resource "aws_sqs_queue" "inference_dlq" {
  name                      = "${local.name_prefix}-inference-dlq"
  message_retention_seconds = var.dlq_retention_seconds

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "event-driven-inference"
  }
}

resource "aws_iam_role" "inference" {
  name = "${local.name_prefix}-inference-lambda"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "sts:AssumeRole"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role" "live_sim" {
  name = "${local.name_prefix}-live-sim-lambda"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "sts:AssumeRole"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_cloudwatch_log_group" "inference" {
  name              = "/aws/lambda/${local.lambda_name}"
  retention_in_days = var.log_retention_days

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "event-driven-inference"
  }
}

resource "aws_cloudwatch_log_group" "live_sim" {
  name              = "/aws/lambda/${local.live_sim_lambda_name}"
  retention_in_days = var.log_retention_days

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "event-driven-live-sim"
  }
}

resource "aws_iam_role_policy" "inference_runtime" {
  name = "${local.name_prefix}-inference-runtime"
  role = aws_iam_role.inference.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadPinnedInferenceInputs"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion"
        ]
        Resource = [
          "${aws_s3_bucket.inference.arn}/inference/requests/*",
          "${aws_s3_bucket.inference.arn}/inference/runs/*/inputs/*",
          "${aws_s3_bucket.inference.arn}/inference/model-bundles/*",
          "${aws_s3_bucket.inference.arn}/inference/runs/*/outputs/*",
          "${aws_s3_bucket.inference.arn}/inference/live-sim/runs/*/outputs/*"
        ]
      },
      {
        Sid    = "WriteInferenceOutputsOnly"
        Effect = "Allow"
        Action = [
          "s3:PutObject"
        ]
        Resource = [
          "${aws_s3_bucket.inference.arn}/inference/runs/*/outputs/*",
          "${aws_s3_bucket.inference.arn}/inference/live-sim/runs/*/outputs/*"
        ]
      },
      {
        Sid    = "WriteLambdaLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "${aws_cloudwatch_log_group.inference.arn}:*"
      },
      {
        Sid      = "SendFailuresToDlq"
        Effect   = "Allow"
        Action   = "sqs:SendMessage"
        Resource = aws_sqs_queue.inference_dlq.arn
      }
    ]
  })
}

resource "aws_iam_role_policy" "live_sim_runtime" {
  name = "${local.name_prefix}-live-sim-runtime"
  role = aws_iam_role.live_sim.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadLiveSimInputsAndState"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion"
        ]
        Resource = [
          "${aws_s3_bucket.inference.arn}/inference/live-sim/runs/*/outputs/*",
          "${aws_s3_bucket.inference.arn}/live-sim/state/*"
        ]
      },
      {
        Sid    = "WriteLiveSimStateAndResults"
        Effect = "Allow"
        Action = [
          "s3:PutObject"
        ]
        Resource = [
          "${aws_s3_bucket.inference.arn}/inference/live-sim/runs/*/outputs/*",
          "${aws_s3_bucket.inference.arn}/live-sim/state/*"
        ]
      },
      {
        Sid    = "WriteLiveSimLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "${aws_cloudwatch_log_group.live_sim.arn}:*"
      },
      {
        Sid      = "SendFailuresToDlq"
        Effect   = "Allow"
        Action   = "sqs:SendMessage"
        Resource = aws_sqs_queue.inference_dlq.arn
      }
    ]
  })
}

resource "aws_lambda_function" "inference" {
  count = var.deploy_lambda ? 1 : 0

  function_name = local.lambda_name
  description   = "Version-pinned active-plus-shadow market-regime inference"
  package_type  = "Image"
  image_uri     = var.lambda_image_uri
  role          = aws_iam_role.inference.arn
  architectures = [var.lambda_architecture]
  memory_size   = var.lambda_memory_mb
  timeout       = var.lambda_timeout_seconds

  reserved_concurrent_executions = 1

  ephemeral_storage {
    size = var.lambda_ephemeral_storage_mb
  }

  environment {
    variables = {
      PYTHONUNBUFFERED = "1"
      LOG_LEVEL        = "INFO"
    }
  }

  tracing_config {
    mode = "PassThrough"
  }

  depends_on = [
    aws_cloudwatch_log_group.inference,
    aws_iam_role_policy.inference_runtime
  ]

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "event-driven-inference"
  }
}

resource "aws_lambda_function" "live_sim" {
  count = var.deploy_lambda && !var.enable_frozen_experiment ? 1 : 0

  function_name = local.live_sim_lambda_name
  description   = "Versioned S3 event-driven paper live simulation"
  package_type  = "Image"
  image_uri     = var.lambda_image_uri
  role          = aws_iam_role.live_sim.arn
  architectures = [var.lambda_architecture]
  memory_size   = var.live_sim_memory_mb
  timeout       = var.live_sim_timeout_seconds

  reserved_concurrent_executions = 1

  ephemeral_storage {
    size = var.lambda_ephemeral_storage_mb
  }

  image_config {
    command = ["src.aws_lambda.live_sim_handler.lambda_handler"]
  }

  environment {
    variables = {
      PYTHONUNBUFFERED = "1"
      LOG_LEVEL        = "INFO"
    }
  }

  tracing_config {
    mode = "PassThrough"
  }

  depends_on = [
    aws_cloudwatch_log_group.live_sim,
    aws_iam_role_policy.live_sim_runtime
  ]

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "event-driven-live-sim"
  }
}

resource "aws_lambda_function_event_invoke_config" "inference" {
  count = var.deploy_lambda ? 1 : 0

  function_name                = aws_lambda_function.inference[0].function_name
  maximum_event_age_in_seconds = 3600
  maximum_retry_attempts       = 1

  destination_config {
    on_failure {
      destination = aws_sqs_queue.inference_dlq.arn
    }
  }
}

resource "aws_lambda_function_event_invoke_config" "live_sim" {
  count = var.deploy_lambda && !var.enable_frozen_experiment ? 1 : 0

  function_name                = aws_lambda_function.live_sim[0].function_name
  maximum_event_age_in_seconds = 3600
  maximum_retry_attempts       = 1

  destination_config {
    on_failure {
      destination = aws_sqs_queue.inference_dlq.arn
    }
  }
}

resource "aws_lambda_permission" "allow_s3_request_events" {
  count = var.deploy_lambda ? 1 : 0

  statement_id   = "AllowS3RequestObjectEvents"
  action         = "lambda:InvokeFunction"
  function_name  = aws_lambda_function.inference[0].function_name
  principal      = "s3.amazonaws.com"
  source_arn     = aws_s3_bucket.inference.arn
  source_account = data.aws_caller_identity.current.account_id
}

resource "aws_lambda_permission" "allow_s3_live_sim_events" {
  count = var.deploy_lambda && !var.enable_frozen_experiment ? 1 : 0

  statement_id   = "AllowS3LiveSimCompletionEvents"
  action         = "lambda:InvokeFunction"
  function_name  = aws_lambda_function.live_sim[0].function_name
  principal      = "s3.amazonaws.com"
  source_arn     = aws_s3_bucket.inference.arn
  source_account = data.aws_caller_identity.current.account_id
}

resource "aws_s3_bucket_notification" "event_routes" {
  count = var.deploy_lambda && !var.enable_frozen_experiment ? 1 : 0

  bucket = aws_s3_bucket.inference.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.inference[0].arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = local.request_prefix
    filter_suffix       = local.request_suffix
  }

  lambda_function {
    lambda_function_arn = aws_lambda_function.live_sim[0].arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = local.live_sim_output_prefix
    filter_suffix       = local.completion_suffix
  }

  depends_on = [
    aws_lambda_permission.allow_s3_request_events,
    aws_lambda_permission.allow_s3_live_sim_events
  ]
}
