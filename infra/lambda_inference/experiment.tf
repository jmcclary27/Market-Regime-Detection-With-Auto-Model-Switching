# Optional frozen-experiment delivery. It is deliberately disabled by default
# so it cannot modify the existing local demo or legacy live-sim path.

locals {
  experiment_lambda_name  = "${local.name_prefix}-frozen-experiment"
  market_prep_lambda_name = "${local.name_prefix}-daily-market-prep"
}

resource "aws_s3_bucket" "dashboard" {
  count         = var.enable_frozen_experiment ? 1 : 0
  bucket_prefix = "${local.name_prefix}-dashboard-"
}

resource "aws_s3_bucket_public_access_block" "dashboard" {
  count                   = var.enable_frozen_experiment ? 1 : 0
  bucket                  = aws_s3_bucket.dashboard[0].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_cloudfront_origin_access_control" "dashboard" {
  count                             = var.enable_frozen_experiment ? 1 : 0
  name                              = "${local.name_prefix}-dashboard"
  description                       = "Read-only frozen experiment dashboard"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

resource "aws_cloudfront_distribution" "dashboard" {
  count               = var.enable_frozen_experiment ? 1 : 0
  enabled             = true
  default_root_object = "index.html"
  price_class         = var.dashboard_price_class

  origin {
    domain_name              = aws_s3_bucket.dashboard[0].bucket_regional_domain_name
    origin_id                = "dashboard"
    origin_access_control_id = aws_cloudfront_origin_access_control.dashboard[0].id
  }

  default_cache_behavior {
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "dashboard"
    viewer_protocol_policy = "redirect-to-https"
    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  viewer_certificate {
    cloudfront_default_certificate = true
  }
}

resource "aws_s3_bucket_policy" "dashboard" {
  count  = var.enable_frozen_experiment ? 1 : 0
  bucket = aws_s3_bucket.dashboard[0].id
  policy = jsonencode({ Version = "2012-10-17", Statement = [{
    Effect    = "Allow", Principal = { Service = "cloudfront.amazonaws.com" }, Action = "s3:GetObject",
    Resource  = "${aws_s3_bucket.dashboard[0].arn}/*",
    Condition = { StringEquals = { "AWS:SourceArn" = aws_cloudfront_distribution.dashboard[0].arn } }
  }] })
}

resource "aws_iam_role" "experiment" {
  count = var.enable_frozen_experiment && var.deploy_lambda ? 1 : 0
  name  = "${local.name_prefix}-frozen-experiment-lambda"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{
    Effect = "Allow", Action = "sts:AssumeRole", Principal = { Service = "lambda.amazonaws.com" }
  }] })
}

resource "aws_iam_role" "market_prep" {
  count = var.enable_frozen_experiment && var.deploy_lambda ? 1 : 0
  name  = "${local.name_prefix}-daily-market-prep-lambda"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{
    Effect = "Allow", Action = "sts:AssumeRole", Principal = { Service = "lambda.amazonaws.com" }
  }] })
}

resource "aws_iam_role_policy" "market_prep" {
  count = var.enable_frozen_experiment && var.deploy_lambda ? 1 : 0
  name  = "${local.name_prefix}-daily-market-prep-runtime"
  role  = aws_iam_role.market_prep[0].id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["s3:GetObject", "s3:GetObjectVersion", "s3:PutObject"], Resource = [
      "${aws_s3_bucket.inference.arn}/experiment/*", "${aws_s3_bucket.inference.arn}/inference/runs/*/inputs/*", "${aws_s3_bucket.inference.arn}/inference/requests/*"
    ] },
    { Effect = "Allow", Action = "secretsmanager:GetSecretValue", Resource = var.alpaca_secret_arn },
    { Effect = "Allow", Action = ["logs:CreateLogStream", "logs:PutLogEvents"], Resource = "*" }
  ] })
}

resource "aws_iam_role_policy" "experiment" {
  count = var.enable_frozen_experiment && var.deploy_lambda ? 1 : 0
  name  = "${local.name_prefix}-frozen-experiment-runtime"
  role  = aws_iam_role.experiment[0].id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["s3:GetObject", "s3:GetObjectVersion", "s3:PutObject"], Resource = [
      "${aws_s3_bucket.inference.arn}/inference/live-sim/runs/*/outputs/*",
      "${aws_s3_bucket.inference.arn}/experiment/*",
      "${aws_s3_bucket.dashboard[0].arn}/*"
    ] },
    { Effect = "Allow", Action = ["logs:CreateLogStream", "logs:PutLogEvents"], Resource = "*" }
  ] })
}

resource "aws_lambda_function" "experiment" {
  count                          = var.enable_frozen_experiment && var.deploy_lambda ? 1 : 0
  function_name                  = local.experiment_lambda_name
  package_type                   = "Image"
  image_uri                      = var.lambda_image_uri
  role                           = aws_iam_role.experiment[0].arn
  architectures                  = [var.lambda_architecture]
  memory_size                    = var.live_sim_memory_mb
  timeout                        = var.live_sim_timeout_seconds
  reserved_concurrent_executions = 1
  image_config { command = ["src.aws_lambda.experiment_handler.lambda_handler"] }
  environment { variables = {
    EXPERIMENT_MANIFEST_KEY     = var.experiment_manifest_key
    EXPERIMENT_DASHBOARD_BUCKET = aws_s3_bucket.dashboard[0].bucket
  } }
  depends_on = [aws_iam_role_policy.experiment]
}

resource "aws_lambda_function" "market_prep" {
  count         = var.enable_frozen_experiment && var.deploy_lambda ? 1 : 0
  function_name = local.market_prep_lambda_name
  package_type  = "Image"
  image_uri     = var.lambda_image_uri
  role          = aws_iam_role.market_prep[0].arn
  architectures = [var.lambda_architecture]
  memory_size   = var.lambda_memory_mb
  timeout       = var.lambda_timeout_seconds
  image_config { command = ["src.aws_lambda.market_preparation_handler.lambda_handler"] }
  environment { variables = {
    EXPERIMENT_BUCKET              = aws_s3_bucket.inference.bucket, EXPERIMENT_MANIFEST_KEY = var.experiment_manifest_key,
    EXPERIMENT_MODEL_BUNDLE_KEY    = var.experiment_model_bundle_key, EXPERIMENT_MODEL_BUNDLE_VERSION_ID = var.experiment_model_bundle_version_id,
    EXPERIMENT_MODEL_BUNDLE_SHA256 = var.experiment_model_bundle_sha256, ALPACA_SECRET_ARN = var.alpaca_secret_arn
  } }
  depends_on = [aws_iam_role_policy.market_prep]
}

resource "aws_iam_role" "scheduler" {
  count = var.enable_frozen_experiment && var.deploy_lambda ? 1 : 0
  name  = "${local.name_prefix}-daily-market-prep-scheduler"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{
    Effect = "Allow", Action = "sts:AssumeRole", Principal = { Service = "scheduler.amazonaws.com" }
  }] })
}

resource "aws_iam_role_policy" "scheduler" {
  count = var.enable_frozen_experiment && var.deploy_lambda ? 1 : 0
  name  = "${local.name_prefix}-daily-market-prep-scheduler"
  role  = aws_iam_role.scheduler[0].id
  policy = jsonencode({ Version = "2012-10-17", Statement = [{
    Effect = "Allow", Action = "lambda:InvokeFunction", Resource = aws_lambda_function.market_prep[0].arn
  }] })
}

resource "aws_scheduler_schedule" "daily_market_prep" {
  count                        = var.enable_frozen_experiment && var.deploy_lambda ? 1 : 0
  name                         = "${local.name_prefix}-daily-market-prep"
  schedule_expression          = "cron(20 16 ? * MON-FRI *)"
  schedule_expression_timezone = "America/New_York"
  flexible_time_window { mode = "OFF" }
  target {
    arn      = aws_lambda_function.market_prep[0].arn
    role_arn = aws_iam_role.scheduler[0].arn
  }
  depends_on = [aws_iam_role_policy.scheduler]
}

resource "aws_lambda_permission" "experiment_s3" {
  count         = var.enable_frozen_experiment && var.deploy_lambda ? 1 : 0
  statement_id  = "AllowS3FrozenExperimentEvents"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.experiment[0].function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.inference.arn
}

# The primary notification is switched atomically between the legacy single
# account and the experiment executor. Both consume the same immutable
# completion contract, so overlapping S3 filters are never configured.
resource "aws_s3_bucket_notification" "experiment_event_routes" {
  count  = var.enable_frozen_experiment && var.deploy_lambda ? 1 : 0
  bucket = aws_s3_bucket.inference.id
  lambda_function {
    lambda_function_arn = aws_lambda_function.inference[0].arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = local.request_prefix
    filter_suffix       = local.request_suffix
  }
  lambda_function {
    lambda_function_arn = aws_lambda_function.experiment[0].arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = local.live_sim_output_prefix
    filter_suffix       = local.completion_suffix
  }
  depends_on = [aws_lambda_permission.allow_s3_request_events, aws_lambda_permission.experiment_s3]
}
