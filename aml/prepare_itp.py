import os
import sys
import azureml.core
from azureml.core import Workspace
from azureml.core import Datastore

# import config_itp_us as config
# import config_itp_sc_hd as config
# import config_itp_sc_yhj as config
# import config_itp_ocr_sc as config
# import config_itp_asia as config
# import config_itp_asia_yhj as config
import config_itp_p40_yhj as config
# import config_itp_rr1_td as config
# import config_itp_sc as config
# import config_itp_ocr_sc as config
# import config_itp_wus2_td as config
# import config_itp_scus_ex as config
# import config_itp_sc_hd as config

subscription_id = config.subscription_id
resource_group = config.resource_group
workspace_name = config.workspace_name

datastore_name = config.datastore_name

blob_container_name = config.blob_container_name
blob_account_name = config.blob_account_name
blob_account_key = config.blob_account_key


# 1. change config_itp_xxx blob info
# 2. import config here, and run this .py file
# 2.5 prepare run...py to what config file you want
# 3. run submit_imagenet_itp.py, change --cluster to what you want, sc, change --model mmseg, change --k to "seg_vit_ssn"
# 4. change exp.sh ROOT to data path
# 

#
# Prepare the workspace.
#
ws = None
try:
    print("Connecting to workspace '%s'..." % workspace_name)
    ws = Workspace(
        subscription_id=subscription_id,
        resource_group=resource_group,
        workspace_name=workspace_name,
    )
except:
    print("Workspace not accessible.")
print(ws.get_details())

ws.write_config()

#
# Register an existing datastore to the workspace.
#
if datastore_name not in ws.datastores:
    Datastore.register_azure_blob_container(
        workspace=ws,
        datastore_name=datastore_name,
        container_name=blob_container_name,
        account_name=blob_account_name,
        account_key=blob_account_key,
    )
    print("Datastore '%s' registered." % datastore_name)
else:
    print("Datastore '%s' has already been regsitered." % datastore_name)

# (END)
# pip install --upgrade azureml-sdk

# pip install --upgrade --disable-pip-version-check --extra-index-url https://azuremlsdktestpypi.azureedge.net/K8s-Compute/D58E86006C65 azureml_contrib_k8s
