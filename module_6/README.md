# HW6: AWS SageMaker Deployment Commands

This file contains the commands I used to deploy the already trained HW5 model as an AWS SageMaker Serverless Inference endpoint

## 1. Prepare model artifact

Package the full MLflow model directory as `model.tar.gz`.

```bash
mkdir -p sagemaker_artifacts

tar -czf sagemaker_artifacts/model.tar.gz -C build/online_model .
```

Check archive contents:

```bash
tar -tzf sagemaker_artifacts/model.tar.gz
```

Expected files:

```text
./MLmodel
./model.pkl
./requirements.txt
./python_env.yaml
./conda.yaml
./input_example.json
./serving_input_example.json
./registered_model_meta
```

## 2. Test SageMaker-compatible container locally

Build local Docker image:

```bash
docker build -f sagemaker.Dockerfile -t wine-sagemaker-local .
```

Run container locally:

```bash
docker run --rm -p 8080:8080 \
  -e MODEL_DIR=/opt/ml/model \
  -v "$(pwd)/build/online_model:/opt/ml/model" \
  wine-sagemaker-local
```

In another terminal, check `/ping`:

```bash
curl -i http://localhost:8080/ping
```

Expected:

```text
HTTP/1.1 200 OK
```

Check `/invocations`:

```bash
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d @build/online_model/serving_input_example.json
```

Expected response format:

```json
{
  "latency_ms": 27.338,
  "n_rows": 3,
  "predictions": [0, 0, 0],
  "success": true
}
```

## 3. Configure AWS variables

```bash
export AWS_REGION=eu-central-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

export ECR_REPO=wine-sagemaker-inference
export IMAGE_TAG=latest
export IMAGE_URI=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG

export S3_BUCKET=wine-sagemaker-hw-$AWS_ACCOUNT_ID
export S3_PREFIX=wine-sagemaker
export MODEL_S3_URI=s3://$S3_BUCKET/$S3_PREFIX/model.tar.gz

export MODEL_NAME=wine-quality-model
export ENDPOINT_CONFIG_NAME=wine-quality-serverless-config
export ENDPOINT_NAME=wine-quality-serverless-endpoint
```

Check variables:

```bash
echo $AWS_ACCOUNT_ID
echo $IMAGE_URI
echo $MODEL_S3_URI
```

## 4. Create S3 bucket and upload model artifact

Create S3 bucket:

```bash
aws s3 mb s3://$S3_BUCKET --region $AWS_REGION
```

Upload model artifact:

```bash
aws s3 cp sagemaker_artifacts/model.tar.gz $MODEL_S3_URI
```

Check uploaded artifact:

```bash
aws s3 ls s3://$S3_BUCKET/$S3_PREFIX/
```

Expected:

```text
model.tar.gz
```

## 5. Create ECR repository

```bash
aws ecr create-repository \
  --repository-name $ECR_REPO \
  --region $AWS_REGION
```

If the repository already exists, continue to the next step.

## 6. Login Docker to ECR

```bash
aws ecr get-login-password --region $AWS_REGION | \
docker login --username AWS --password-stdin \
$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
```

Expected:

```text
Login Succeeded
```

## 7. Build and push image to ECR

Use `buildx` with Docker-compatible settings for SageMaker.

```bash
docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  -f sagemaker.Dockerfile \
  -t $IMAGE_URI \
  --push \
  .
```

The options `--provenance=false` and `--sbom=false` are used to avoid pushing an OCI image index that SageMaker may reject.

## 8. Create SageMaker execution role

The role used in this homework:

```bash
export SAGEMAKER_ROLE_ARN=$(aws iam get-role \
  --role-name SageMakerWineExecutionRole \
  --query 'Role.Arn' \
  --output text)

echo $SAGEMAKER_ROLE_ARN
```

Expected format:

```text
arn:aws:iam::<account-id>:role/SageMakerWineExecutionRole
```

## 9. Create SageMaker model

```bash
aws sagemaker create-model \
  --region $AWS_REGION \
  --model-name $MODEL_NAME \
  --execution-role-arn $SAGEMAKER_ROLE_ARN \
  --primary-container Image=$IMAGE_URI,ModelDataUrl=$MODEL_S3_URI,Mode=SingleModel
```

Expected output:

```json
{
  "ModelArn": "arn:aws:sagemaker:eu-central-1:<account-id>:model/wine-quality-model"
}
```

## 10. Create serverless endpoint configuration

```bash
aws sagemaker create-endpoint-config \
  --region $AWS_REGION \
  --endpoint-config-name $ENDPOINT_CONFIG_NAME \
  --production-variants "[
    {
      \"VariantName\": \"AllTraffic\",
      \"ModelName\": \"$MODEL_NAME\",
      \"ServerlessConfig\": {
        \"MemorySizeInMB\": 2048,
        \"MaxConcurrency\": 1
      }
    }
  ]"
```

Expected output:

```json
{
  "EndpointConfigArn": "arn:aws:sagemaker:eu-central-1:<account-id>:endpoint-config/wine-quality-serverless-config"
}
```

## 11. Create SageMaker endpoint

```bash
aws sagemaker create-endpoint \
  --region $AWS_REGION \
  --endpoint-name $ENDPOINT_NAME \
  --endpoint-config-name $ENDPOINT_CONFIG_NAME
```

Expected output:

```json
{
  "EndpointArn": "arn:aws:sagemaker:eu-central-1:<account-id>:endpoint/wine-quality-serverless-endpoint"
}
```

## 12. Check endpoint status

```bash
aws sagemaker describe-endpoint \
  --region $AWS_REGION \
  --endpoint-name $ENDPOINT_NAME \
  --query "EndpointStatus"
```

Wait approx. 15 minutes until the status becomes:

```text
"InService"
```

Full endpoint description:

```bash
aws sagemaker describe-endpoint \
  --region $AWS_REGION \
  --endpoint-name $ENDPOINT_NAME
```

## 13. Invoke deployed endpoint

Invoke endpoint with the model input example:

```bash
aws sagemaker-runtime invoke-endpoint \
  --region $AWS_REGION \
  --endpoint-name $ENDPOINT_NAME \
  --content-type application/json \
  --body fileb://build/online_model/serving_input_example.json \
  response.json
```

Expected metadata output:

```json
{
  "ContentType": "application/json",
  "InvokedProductionVariant": "AllTraffic"
}
```

Check prediction result:

```bash
cat response.json
```

Example response:

```json
{
  "latency_ms": 5.746,
  "n_rows": 3,
  "predictions": [0, 0, 0],
  "success": true
}
```

This confirms that the deployed SageMaker endpoint successfully returned predictions.

## 14. Optional: check endpoint logs

List log groups:

```bash
aws logs describe-log-groups \
  --region $AWS_REGION \
  --log-group-name-prefix "/aws/sagemaker/Endpoints/$ENDPOINT_NAME"
```

Tail endpoint logs:

```bash
aws logs tail "/aws/sagemaker/Endpoints/$ENDPOINT_NAME" \
  --region $AWS_REGION \
  --since 30m
```

## 15. Cleanup resources

Run cleanup after the demo to avoid extra charges.

Delete endpoint:

```bash
aws sagemaker delete-endpoint \
  --region $AWS_REGION \
  --endpoint-name $ENDPOINT_NAME
```

Delete endpoint configuration:

```bash
aws sagemaker delete-endpoint-config \
  --region $AWS_REGION \
  --endpoint-config-name $ENDPOINT_CONFIG_NAME
```

Delete model:

```bash
aws sagemaker delete-model \
  --region $AWS_REGION \
  --model-name $MODEL_NAME
```

Delete ECR repository and image:

```bash
aws ecr delete-repository \
  --region $AWS_REGION \
  --repository-name $ECR_REPO \
  --force
```

Delete model artifact from S3:

```bash
aws s3 rm $MODEL_S3_URI
```

Delete empty S3 bucket:

```bash
aws s3 rb s3://$S3_BUCKET
```

## 16. Verification after cleanup

Check that endpoint no longer exists:

```bash
aws sagemaker describe-endpoint \
  --region $AWS_REGION \
  --endpoint-name $ENDPOINT_NAME
```

Expected: endpoint not found.

Check that ECR repository no longer exists:

```bash
aws ecr describe-repositories \
  --region $AWS_REGION \
  --repository-names $ECR_REPO
```

Expected: repository not found.

Check that S3 bucket was removed:

```bash
aws s3 ls s3://$S3_BUCKET
```

Expected: bucket not found.

## Screenshots

Screenshots for the demo are stored in:

```text
screenshots/
```

Recommended screenshot order:

```text
01_container_logs.png
01_healthcheck_ping.png
01_invocations_local_prediction.png

02_ecr_repo_image_latest.png

03_trained_model_artifact_to_S3.png
03_sagemaker_model.png
03_endpoint_configuration.png
03_endpoint_inservice.png

04_invoke_endpoint_command.png
04_predictions_response_json.png
```