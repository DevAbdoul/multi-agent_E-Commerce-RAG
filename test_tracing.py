import sys
import os
import json

# Add parent directory to path so config can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import boto3
import config

region = getattr(config, 'AWS_REGION', 'us-east-1')
runtime_arn = getattr(config, 'AGENTCORE_RUNTIME_ARN', '')

print(f"Targeting AgentCore Runtime: {runtime_arn}")

client = boto3.client('bedrock-agentcore', region_name=region)

# Prepare JSON payload matching the AgentCore runtime interface
payload_data = json.dumps({
    "prompt": "I need help returning an order."
})

try:
    response = client.invoke_agent_runtime(
        agentRuntimeArn=runtime_arn,
        payload=payload_data.encode('utf-8') if isinstance(payload_data, str) else payload_data
    )

    print("\nInvocation successful! CloudWatch telemetry & X-Ray traces generated.")
    print("Response payload:", response.get('payload', {}).read().decode('utf-8') if 'payload' in response else response)

except Exception as e:
    print(f"\nAPI execution error: {e}")