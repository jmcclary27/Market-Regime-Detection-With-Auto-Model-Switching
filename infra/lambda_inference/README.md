# Event-driven inference and paper live-simulation infrastructure

This isolated Terraform root provisions the low-cost event-driven inference
and paper live-simulation path. It does not modify or reuse the existing EC2
stack in `infra/`.

By default `deploy_lambda=false`, so an initial Terraform apply can create the
versioned S3 bucket, ECR repository, IAM roles, log groups, and failure DLQ
without creating either Lambda function or an event notification. No command
in this repository automatically reads `.env` or deploys this stack.

## Bootstrap after review

1. Run Terraform here with `deploy_lambda=false`, then get the ECR URL from
   `terraform output -raw ecr_repository_url`.
2. Build and push the dedicated image for the selected architecture. The
   default is `arm64`, which costs less at runtime:

   ```bash
   docker buildx build --platform linux/arm64 --push \
     -f docker/lambda-inference/Dockerfile \
     -t <ecr-url>:<immutable-tag> .
   ```

3. Resolve the pushed image digest and apply again with
   `-var='deploy_lambda=true'` and
   `-var='lambda_image_uri=<ecr-url>@sha256:<digest>'`. A digest, not a mutable
   tag, is the intended deployment reference. This deploys the inference and
   paper live-simulation handlers from the same image with different commands.

4. Run one non-production `batch` request and then one `live_sim` request
   through the contract in
   [`docs/aws_lambda_inference.md`](../../docs/aws_lambda_inference.md). Check
   both completion objects and the live-sim result before enabling a scheduled
   producer.

The Lambdas are deliberately not attached to a VPC. They need no private
network resources, and a VPC/NAT gateway would add a recurring cost. Both use
their own IAM execution roles for S3 and CloudWatch access; do not put AWS keys
in Lambda environment variables.
