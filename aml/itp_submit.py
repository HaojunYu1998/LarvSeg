from azureml.core import Workspace
from azureml.core import Datastore
from azureml.core import Experiment
from azureml.core.compute import ComputeTarget
from azureml.train.dnn import PyTorch
from azureml.train.dnn import TensorFlow
from azureml.core.compute import AksCompute

ws = Workspace.from_config()
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

cluster_name = "itpseasiav100cl"
datastore_name = "wenxie_msraimseasia"

from azureml.core.compute import ComputeTarget
from azureml.contrib.core.compute.itpcompute import ItpCompute
for key, target in ws.compute_targets.items():
    # if type(target) is AksCompute:
    print('Found compute target:{}\ttype:{}\tprovisioning_state:{}\tlocation:{}'.format(target.name, target.type, target.provisioning_state, target.location))

ct = ComputeTarget(workspace=ws, name=cluster_name)
ds = Datastore(workspace=ws, name=datastore_name)

# use get_status() to get a detailed status for the current cluster. 
print(ct.get_status())

script_params = {
    '--rootpath': ds.path('wenxie/GCR').as_mount()
}

est = TensorFlow(source_directory='.',
                 entry_script='entry.py',
                 script_params=script_params,
                 compute_target=ct,
                 use_gpu=True, 
                 shm_size='256G',
                 custom_docker_image="wenxuanxie/tensorflow:1.10.1-py35",
                 user_managed=True)

from azureml.contrib.core.itprunconfig import ItpComputeConfiguration

exp = Experiment(workspace=ws, name='mnist-itp')

compute_config = ItpComputeConfiguration()
compute_config.configuration = {
    'gpu_count': 1
}
est.run_config.cmk8scompute = compute_config
run = exp.submit(est)
