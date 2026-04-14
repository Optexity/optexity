from typing import NewType

TaskID = NewType("TaskID", str)
DedupKey = NewType("DedupKey", str)
UserID = NewType("UserID", str)
CompanyID = NewType("CompanyID", str)
RecordingID = NewType("RecordingID", str)

ChildArn = NewType("ChildArn", str)
ChildUrl = NewType("ChildUrl", str)
CleanedUrl = NewType("CleanedUrl", str)
EC2PrivateIP = NewType("EC2PrivateIP", str)
