import base64
import os
import paramiko
from os.path import basename

key = paramiko.RSAKey.from_private_key_file("/Users/qianminming/GoogleDrive/PhD/aws/qianminming_ami_test.pem")
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print("connecting")
client.connect( hostname = "ec2-13-55-93-184.ap-southeast-2.compute.amazonaws.com", username = "ec2-user", pkey = key )
print("connected")
sftp = client.open_sftp()

report_folder = input("reportname: ")
local_path = "/Users/qianminming/GoogleDrive/PhD/DeepSet/reports"
os.mkdir("/".join([local_path, report_folder]))
remote_files = [
    "/home/ec2-user/vgg16_finetune_mutli_label/losses.png",
    "/home/ec2-user/vgg16_finetune_mutli_label/train_precision_recall.png",
    "/home/ec2-user/vgg16_finetune_mutli_label/val_precision_recall.png",
    "/home/ec2-user/vgg16_finetune_mutli_label/nohup.out",
]

for remote_file in remote_files:
    local_file = "/".join([local_path, report_folder, basename(remote_file)])
    sftp.get( remote_file, local_file)

client.close()