#!/usr/bin/env bash
set -euo pipefail
OUT=ecs_diagnostics
mkdir -p "$OUT"
REGION="eu-north-1"
CLUSTER="OCR"
SERVICE="ocr-service-qu9kzscy"
TASK_DEF="ocr:3"
ECR_REPO="zouheir/sof-ocr"
ENI="eni-06cf50452bf3147f3"
SUBNET="subnet-0a297c25f1fe3e612"
SG="sg-0923c68c0c42bb5fb"
PROFILE="temp"

aws --profile "$PROFILE" ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --region "$REGION" > "$OUT/describe-services.json" || true
aws --profile "$PROFILE" ecs describe-task-definition --task-definition "$TASK_DEF" --region "$REGION" > "$OUT/describe-task-definition.json" || true
aws --profile "$PROFILE" ecr describe-images --repository-name "$ECR_REPO" --region "$REGION" > "$OUT/ecr-describe-images.json" || true
aws --profile "$PROFILE" ec2 describe-network-interfaces --network-interface-ids "$ENI" --region "$REGION" > "$OUT/describe-eni.json" || true
aws --profile "$PROFILE" ec2 describe-subnets --subnet-ids "$SUBNET" --region "$REGION" > "$OUT/describe-subnet.json" || true
aws --profile "$PROFILE" ec2 describe-route-tables --filters "Name=association.subnet-id,Values=$SUBNET" --region "$REGION" > "$OUT/describe-route-tables.json" || true
aws --profile "$PROFILE" ec2 describe-security-groups --group-ids "$SG" --region "$REGION" > "$OUT/describe-sg.json" || true
aws --profile "$PROFILE" ecs list-tasks --cluster "$CLUSTER" --region "$REGION" --max-items 50 > "$OUT/list-tasks.json" || true
TASKS=$(jq -r '.taskArns[]' "$OUT/list-tasks.json" 2>/dev/null || true)
if [ -n "$TASKS" ]; then
  aws --profile "$PROFILE" ecs describe-tasks --cluster "$CLUSTER" --tasks $TASKS --region "$REGION" > "$OUT/describe-tasks.json" || true
fi
aws --profile "$PROFILE" logs describe-log-groups --log-group-name-prefix "/ecs" --region "$REGION" > "$OUT/log-groups.json" || true

echo "Saved diagnostics in $OUT"
