import boto3
from botocore.exceptions import ClientError

# Replace these with your info
SENDER = ""  # Must be a verified email in SES
RECIPIENT = ""  # Can be unverified if your account is out of sandbox
AWS_REGION = "us-east-1"  # SES region, e.g. us-east-1
SUBJECT = "Test Email from AWS SES"
BODY_TEXT = ("Hello,\r\n"
             "This is a test email sent through AWS SES using boto3 in Python.")
BODY_HTML = """<html>
<head></head>
<body>
  <h1>Hello!</h1>
  <p>This email was sent with <a href='https://aws.amazon.com/ses/'>AWS SES</a> using <b>boto3</b> in Python.</p>
</body>
</html>
            """
CHARSET = "UTF-8"

def send_email():
    # Create a new SES client
    client = boto3.client(
        'ses', 
        region_name=AWS_REGION,
        aws_access_key_id="",
        aws_secret_access_key=""
    )

    # Try to send the email
    try:
        response = client.send_email(
            Destination={
                'ToAddresses': [
                    RECIPIENT,
                ],
            },
            Message={
                'Body': {
                    'Html': {
                        'Charset': CHARSET,
                        'Data': BODY_HTML,
                    },
                    'Text': {
                        'Charset': CHARSET,
                        'Data': BODY_TEXT,
                    },
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': SUBJECT,
                },
            },
            Source=SENDER,
        )
    except ClientError as e:
        print(f"Error sending email: {e.response['Error']['Message']}")
    else:
        print(f"Email sent! Message ID: {response['MessageId']}")

if __name__ == "__main__":
    send_email()