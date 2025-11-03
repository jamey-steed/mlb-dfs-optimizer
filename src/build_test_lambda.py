import shutil
import uuid
import boto3
import json
import botocore
import requests
import pickle


def build_from_lambda(
    NUM_LINEUPS, pitcher_targets, team_targets, player_metadata, seperate=False
):

    ownership_targets = {"stack_team": team_targets, "pitcher": pitcher_targets}
    config = botocore.config.Config(
        read_timeout=900, connect_timeout=900, retries={"max_attempts": 0}
    )
    print("building lambda client")
    lambda_client = boto3.client("lambda", config=config)

    job_id = str(uuid.uuid4())
    print("calling lambda")
    if not seperate:
        per_lambda = int(NUM_LINEUPS // 200) + 1
        payload = {
            "job_id": job_id,
            "player_metadata": player_metadata,
            "ownership_targets": ownership_targets,
            "total_lineups": NUM_LINEUPS,
            "lineups_per_lambda": per_lambda,  # optional, default is 100
        }

        response = lambda_client.invoke(
            FunctionName="fanout_lambda",
            InvocationType="RequestResponse",
            Payload=json.dumps(payload).encode("utf-8"),
        )
        result = json.loads(response["Payload"].read())
        result_body = json.loads(result["body"])
        print("result received")
        if "lineups" in result_body:
            lineups = result_body["lineups"]
            import pickle

            with open("output_data/field_lineups.pkl", "wb") as f:
                pickle.dump(lineups, f)
        elif "presigned_url" in result_body:
            print("calling s3")

            url = result_body["presigned_url"]
            response = requests.get(url, stream=True)
            with open("output_data/field_lineups.pkl", "wb") as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)
            with open("output_data/field_lineups.pkl", "rb") as f:
                lineups = pickle.load(f)["lineups"]
            with open("output_data/field_lineups.pkl", "wb") as f:
                pickle.dump(lineups, f)
            # print(lineups)
        return lineups
    else:
        per_lambda = int(30000 / 200)
        payload = {
            "job_id": job_id,
            "player_metadata": player_metadata,
            "ownership_targets": ownership_targets,
            "total_lineups": 30000,
            "lineups_per_lambda": per_lambda,  # optional, default is 100
            "seperate": True,
            "field_size": NUM_LINEUPS,
        }
        response = lambda_client.invoke(
            FunctionName="fanout_lambda",  # Replace with actual name
            InvocationType="RequestResponse",
            Payload=json.dumps(payload).encode("utf-8"),
        )
        result = json.loads(response["Payload"].read())
        result_body = json.loads(result["body"])
        print("result received")
        if "field_lineups" in result_body:
            field_lineups = result_body["field_lineups"]
            remaining_lineups = result_body["remaining_lineups"]
            import pickle

            with open("output_data/field_lineups.pkl", "wb") as f:
                pickle.dump(field_lineups, f)
            with open("remaining_lineups.pkl", "wb") as f:
                pickle.dump(remaining_lineups, f)
        elif "presigned_url" in result_body:
            print("calling s3")

            url = result_body["presigned_url"]
            response = requests.get(url, stream=True)
            with open("output_data/field_lineups.pkl", "wb") as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)
            with open("output_data/field_lineups.pkl", "rb") as f:
                lineups = pickle.load(f)["lineups"]
            with open("output_data/field_lineups.pkl", "wb") as f:
                pickle.dump(lineups, f)
            # print(lineups)
        return field_lineups, remaining_lineups
