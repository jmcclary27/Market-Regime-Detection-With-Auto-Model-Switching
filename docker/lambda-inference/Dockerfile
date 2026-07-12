# Dedicated runtime image for the event-driven inference and paper live-sim Lambdas.
# Model artifacts are deliberately not baked into this image. Each request
# downloads a version-pinned model bundle from S3 so promotion is reproducible.
FROM public.ecr.aws/lambda/python:3.11

COPY docker/lambda-inference/requirements-lambda.txt ${LAMBDA_TASK_ROOT}/requirements-lambda.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt

COPY src ${LAMBDA_TASK_ROOT}/src

CMD ["src.aws_lambda.inference_handler.lambda_handler"]
