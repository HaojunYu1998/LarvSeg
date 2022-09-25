import os
import sys
import pprint
import argparse
import azureml.core
from azureml.core import Workspace
from azureml.core import Datastore
from azureml.core import Experiment
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import ContainerRegistry
from azureml.train.dnn import PyTorch
from azureml.train.dnn import Nccl
from azureml.train.estimator import Estimator
from azureml.core.runconfig import MpiConfiguration

from azureml.contrib.core.k8srunconfig import K8sComputeConfiguration


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AML Generic Launcher")
    parser.add_argument("--cfg", default="")
    parser.add_argument(
        "--status", default="train", help="The status of the job, e.g., train or test"
    )
    parser.add_argument("--flag", default="", help="The suffix string to mark the job.")
    parser.add_argument(
        "--num_gpus", default=8, type=int, help="The number of gpus for training."
    )
    parser.add_argument("--batch_size", default=64, help="The batch size in each GPU.")
    parser.add_argument("--k", default="10", 
                        help="Top K classes")      
    parser.add_argument("--d", default="10", 
                        help="Hidden dimension") 
    parser.add_argument("--img_loss_weight", default="1.0", 
                        help="Image loss weight") 
    parser.add_argument(
        "--dataset", default="", help="The dataset to run the job."
    )
    parser.add_argument("--model", default="hrnet")
    parser.add_argument("--cluster", default="ocr_sc")  # rr1 sc ocr_wus ocr_sc
    parser.add_argument(
        "--preemp", action="store_true", dest="preemp", help="Use preemptible GPUs."
    )
    parser.add_argument("--nnode", type=int, default=1)
    args, _ = parser.parse_known_args()

    if args.cluster == "us":
        import config_itp_us as config
    elif args.cluster == "us2":
        import config_itp_us2 as config
    elif args.cluster == "eus":
        import config_itp_eus as config
    elif args.cluster == "sc":
        import config_itp_sc as config
    elif args.cluster == "sc1":
        import config_itp_sc_hd as config
    elif args.cluster == "sc2":
        import config_itp_sc_yhj as config
    elif args.cluster == "scus_ex":
        import config_itp_scus_ex as config
    elif args.cluster == "scus":
        import config_itp_scus as config
    elif args.cluster == "asia":
        import config_itp_asia as config
    elif args.cluster == "asia1":
        import config_itp_asia_yhj as config
    elif args.cluster == "rr1":
        import config_itp_rr1 as config
    elif args.cluster == "rr1_td":
        import config_itp_rr1_td as config
    elif args.cluster == "rr1_haodi":
        import config_itp_rr1_haodi as config
    elif args.cluster == "rr1_yhj":
        import config_itp_rr1_yhj as config
    elif args.cluster == "wus2":
        import config_itp_wus2 as config
    elif args.cluster == "wus":
        import config_itp_wus2_hd as config
    elif args.cluster == "wus2_td":
        import config_itp_wus2_td as config
    elif args.cluster == "ocr_wus":
        import config_itp_ocr_wus as config
    elif args.cluster == "ocr_sc":
        import config_itp_ocr_sc as config
    elif args.cluster == "itp_ocr_res":
        import config_vision_itp_ocr_res as config
    elif args.cluster == "p40":
        import config_itp_p40_yhj as config
    else:
        raise ValueError("Invalid Cluster Name: {}".format(args.cluster))

    # docker image registry, no need to change if you want to use Philly docker
    # container_registry_address = "phillyregistry.azurecr.io/" # example : "phillyregistry.azurecr.io"
    container_registry_address = "docker.io/"  # example : "phillyregistry.azurecr.io"
    container_registry_username = ""
    container_registry_password = ""
    # custom_docker_image="deppmeng/pytorch:pytorch1.7-HRNetCls-mpi"
    # freddierao/pytorch1.7-hrnetcls-mpi-apex:v5
    if "mmseg-vit" in args.model:
        custom_docker_image = "hsfzxjy/mmseg:pytorch1.8.1-cuda10.2-cudnn7-devel"
    elif "mformer2" == args.model:
        custom_docker_image = "hardyhe/mmseg:pytorch1.8.1-cuda10.2-cudnn7-devel_1203"
    elif "mmseg" in args.model or "ocr" in args.model or "detr" in args.model or "mformer" in args.model:
        custom_docker_image = "hardyhe/mmseg:pytorch1.8.1-cuda10.2-cudnn7-devel_1125"
    else:
        custom_docker_image = "dylan85851/pytorch1.9:20210918_v8"

    source_directory = "./docker"
    

    subscription_id = config.subscription_id
    resource_group = config.resource_group
    workspace_name = config.workspace_name
    ws = Workspace(
        subscription_id=subscription_id,
        resource_group=resource_group,
        workspace_name=workspace_name,
    )

    cluster_name = config.cluster_name
    ct = ComputeTarget(workspace=ws, name=cluster_name)
    datastore_name = config.datastore_name
    ds = Datastore(workspace=ws, name=datastore_name)

    if args.dataset == "cityscapes":
        script_params = {
            "--code_dir": ds.path("haodi/code/imagenet/HRT-Seg").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
        dataset = "cityscapes"
    elif args.dataset == "cityscapes-topk":
        script_params = {
            "--code_dir": ds.path("haodi/code/imagenet/HRT-Seg").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
            "--k": args.k,
            "--d": args.d,
        }
        dataset = "cityscapes_topk"
    elif args.dataset == "coco-stuff":
        script_params = {
            "--code_dir": ds.path("haodi/code/imagenet/HRT-Seg").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
        dataset = "coco-stuff"
    elif args.dataset == "coco-stuff-topk":
        script_params = {
            "--code_dir": ds.path("haodi/code/imagenet/HRT-Seg").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
            "--k": args.k,
            "--d": args.d,
        }
        dataset = "coco-stuff_topk"
    elif args.dataset == "ade20k":
        script_params = {
            "--code_dir": ds.path("haodi/code/imagenet/HRT-Seg").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
        dataset = "ade20k"
    elif args.dataset == "ade20k-topk":
        script_params = {
            "--code_dir": ds.path("haodi/code/imagenet/HRT-Seg").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
            "--k": args.k,
            "--d": args.d,
        }
        dataset = "ade20k_topk"
    elif args.dataset == "adefull":
        script_params = {
            "--code_dir": ds.path("haodi/code/imagenet/HRT-Seg").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
        dataset = "adefull"
    elif args.dataset == "lvis":
        script_params = {
            "--code_dir": ds.path("haodi/code/imagenet/HRT-Seg").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
        dataset = "lvis"
    elif args.dataset == "lvis-topk":
        script_params = {
            "--code_dir": ds.path("haodi/code/imagenet/HRT-Seg").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
            "--k": args.k,
            "--d": args.d,
        }
        dataset = "lvis_topk"
    elif args.model == "mmseg":
        script_params = {
            "--code_dir": ds.path("mmseg").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
            "--k": args.k,
        }
    elif args.model == "mmseg1":
        script_params = {
            "--code_dir": ds.path("haodi/code/imagenet/mmseg-msravc").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }    
    elif args.model == "detr":
        script_params = {
            "--code_dir": ds.path("haodi/code/imagenet/Deformable-DETR-MLSeg").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mformer":
        script_params = {
            "--code_dir": ds.path("haodi/code/imagenet/Maskformer-main").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mformer2":
        script_params = {
            "--code_dir": ds.path("haodi/code/imagenet/Mask2Former").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "q2l":
        script_params = {
            "--code_dir": ds.path("haodi/code/imagenet/query2labels").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg_2":
        script_params = {
            "--code_dir": ds.path("haodi/code/imagenet/mmseg_backup/mmseg").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "beit":
        script_params = {
            "--code_dir": ds.path("luoxiao/unilm/beit").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mocov3_stage2":
        script_params = {
            "--code_dir": ds.path("luoxiao/yude/ssl-seg-finetune/mocov3").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mocov3_stage1":
            script_params = {
            "--code_dir": ds.path("luoxiao/yude/ssl-seg-finetune/mocov3").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mocov3_stage3":
            script_params = {
            "--code_dir": ds.path("luoxiao/yude/ssl-seg-finetune/semantic_segmentation").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "beit_mask":
            script_params = {
            "--code_dir": ds.path("luoxiao/yude/ssl-seg-finetune/semantic_segmentation").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mlm_ade_training":
            script_params = {
            "--code_dir": ds.path("luoxiao/yude/ssl-seg-finetune/beit").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mlmbeit_fine":
            script_params = {
            "--code_dir": ds.path("luoxiao/yude/ssl-seg-finetune/semantic_segmentation").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "beit_stage2":
        script_params = {
            "--code_dir": ds.path("luoxiao/ssl-seg-finetune/beit").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "beit_stage3":
        script_params = {
            "--code_dir": ds.path("luoxiao/ssl-seg-finetune/beit").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "beit_mask_fin":
        script_params = {
            "--code_dir": ds.path("luoxiao/yude/ssl-seg-finetune/semantic_segmentation").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "beit_mask_fine":
        script_params = {
            "--code_dir": ds.path("luoxiao/yude/ssl-seg-finetune/semantic_segmentation").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-ssn":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-ssn-1":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-ssn-2":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-ssn-pnp-large":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-ssn-pnp-small":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-ssn-vitt":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-ssn-vit-hi-mha":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-ssn-vit-hi-mha_unrefine":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-ssn-vit-hi-mha_4_8" or args.model == "mmseg-ssn-vit-hi-mha_4_8_10":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-ssn-vit-hi-mha_4_10":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-ssn-vit-hi-mha_4_8_unrefine" or args.model == "mmseg-ssn-vit-hi-mha_4_8_10_unrefine" :
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-ssn-vit-hi-mha_4_10_unrefine":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-ssn-vit-hi-mha-p":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-ssn-vit-hi-mha-p_unrefine":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-ssn-vit-hi-mha-p_4_8" or args.model == "mmseg-ssn-vit-hi-mha-p_4_8_10" :
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-ssn-vit-hi-mha-p_4_10":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-ssn-vit-hi-mha-p_4_8_unrefine" or args.model =="mmseg-ssn-vit-hi-mha-p_4_8_10_unrefine":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-ssn-vit-hi-mha-p_4_10_unrefine":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-ssn-vit-p-rhi-mha" or args.model == "mmseg-ssn-vit-p-rdhi-mha":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-ssn-vit-p-rdfhi-mha" or args.model == "mmseg-ssn-vit-p-rdfhi-mha_o":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-ssn-vit-p-rdfhi-mha_1" or args.model == "mmseg-ssn-vit-p-rdfhi-mha_o_1":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-ssn-vit-p-rdfhi-mha_o_2" or args.model == "mmseg-ssn-vit-p-rdfhi-mha_o_3" or args.model == "mmseg-ssn-vit-p-rdfhi-mha_o_4":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }



    elif args.model == "mmseg-ssn-vit-p-hi-mha":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-ssn-vitp":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-vit_ori":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }

    elif args.model == "mmseg-ssn-vit-u_2_4_8_10" or args.model == "mmseg-ssn-vit-u_3_5_8_11" or args.model == "mmseg-ssn-vit-u_4_9" or args.model == "mmseg-ssn-vit-u_5_10":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }

    elif args.model == "mmseg-ssn-vit-u_2_4_8_10_s" or args.model == "mmseg-ssn-vit-u_3_5_8_11_s" or args.model == "mmseg-ssn-vit-u_4_9_s" or args.model == "mmseg-ssn-vit-u_5_10_s":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }

    elif args.model == "mmseg-ssn-vit-u_2_4_8_10_unrefine" or args.model == "mmseg-ssn-vit-u_3_5_8_11_unrefine" or args.model == "mmseg-ssn-vit-u_4_9_unrefine" or args.model == "mmseg-ssn-vit-u_5_10_unrefine":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-vit_ssn_SC" or args.model == "mmseg-vit_ada_D"or args.model == "mmseg-vit_Q" or args.model == "mmseg-vit_ada_HD" or args.model == "mmseg-vit_ssn" or args.model == "mmseg-vit_ssn_L" or args.model == "mmseg-vit_ssn_S" or args.model == "mmseg-vit_ssn_S_topk_M" or args.model == "mmseg-vit_ssn_S_topk":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }
    elif args.model == "mmseg-vit_ssn_C_topk_ade20k_3" or  args.model == "mmseg-vit_ssn_C_topk_3" or args.model == "mmseg-vit_ssn_C_topk_ade20k_2" or  args.model == "mmseg-vit_ssn_C_topk_2" or args.model == "mmseg-vit_ssn_C_topk_ade20k" or  args.model == "mmseg-vit_ssn_C_topk" or  args.model == "mmseg-vit_ssn_H_ade20k" or args.model == "mmseg-vit_ssn_H" or args.model == "mmseg-vit_ssn_H_topk_ade20k" or args.model == "mmseg-vit_ssn_H_topk" or args.model == "mmseg-vit_ssn_9_11_S_topk" or args.model == "mmseg-vit_ssn_9_11_S_topk_ade20k" or args.model == "mmseg-vit_ssn_8_11_S_topk" or args.model == "mmseg-vit_ssn_8_11_S_topk_ade20k" or args.model == "mmseg-vit_ssn_6_8_S_topk" or args.model == "mmseg-vit_ssn_6_8_S_topk_ade20k" or args.model == "mmseg-vit_ssn_SC_ade20k" or  args.model == "mmseg-vit_ssn_6_7_S_topk" or  args.model == "mmseg-vit_ssn_6_7_S_topk_ade20k" or  args.model == "mmseg-vit_ssn_8_9_S_topk_ade20k" or  args.model == "mmseg-vit_ssn_8_9_S_topk"  or  args.model == "mmseg-vit_ssn_10_S_topk" or  args.model == "mmseg-vit_ssn_10_S_topk_ade20k"  or args.model == "mmseg-vit_ssn_S_topk_offset"  or args.model == "mmseg-vit_ssn_S_topk_offset_ade20k" or args.model == "mmseg-vit_ori_ade20k"  or args.model == "mmseg-vit_ssn_ade20k" or args.model == "mmseg-vit_ssn_S_ade20k" or args.model == "mmseg-vit_ssn_S_topk_M_ade20k" or args.model == "mmseg-vit_ssn_S_topk_ade20k" or args.model == "mmseg-vit_Q_ade20k":
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }

    elif "mmseg-vit_ssn" in args.model or "mmseg-vit_ssn_CH" in args.model:
        script_params = {
            "--code_dir": ds.path("luoxiao/code/imagenet/mmseg_ssn").as_mount(),
            "--data_dir": ds.path("dataset").as_mount(),
            "--num_gpus": args.num_gpus,
            "--batch_size": args.batch_size,
            "--cluster": args.cluster,
        }

    else:
        raise ValueError("Invalid dataset: {}".format(args.dataset))
    
    if "img" in args.model:
        script_params["--img_loss_weight"]=args.img_loss_weight

    if ("mmseg" in args.model or "q2l" == args.model) and "mmseg-" not in args.model:
        if args.k != "10":
            entry_script = "run_{}_{}.py".format(args.model, args.k)
        else:
            entry_script = "run_mmseg.py"
    elif args.model == "detr":
        entry_script = "run_{}_{}.py".format(args.model, args.k)
    elif args.model == "mformer":
        entry_script = "run_{}_{}.py".format(args.model, args.k)
    elif args.model == "mformer2":
        entry_script = "run_{}_{}.py".format(args.model, args.k)
    elif args.model == "beit_stage2":
        entry_script = "run_beit_stage2.py"
    elif args.model == "beit_stage3":
        entry_script = "run_beit_stage3.py"
    elif args.model == "mocov3_stage2":
        entry_script = "run_mocov3_stage2.py"
    elif args.model == "mocov3_stage1":
        entry_script = "run_mocov3_stage1.py"
    elif args.model == "mocov3_stage3":
        entry_script = "run_mocov3_stage3.py"
    elif args.model == "beit_mask":
        entry_script = "run_beit_mask.py"
    elif args.model == "mlmbeit_fine":
        entry_script = "run_beit_mlmbeit_fine.py"
    elif args.model == "mmseg-ssn":
        entry_script = "run_mmseg-ssn.py"
    elif args.model == "mmseg-ssn-1":
        entry_script = "run_mmseg-ssn-1.py"
    elif args.model == "mmseg-ssn-2":
        entry_script = "run_mmseg-ssn-2.py"
    elif args.model == "mmseg-ssn-pnp-large":
        entry_script = "run_mmseg-ssn-pnp-large.py"
    elif args.model == "mmseg-ssn-pnp-small":
        entry_script = "run_mmseg-ssn-pnp-small.py"
    elif args.model == "mmseg-ssn-vitt":
        entry_script = "run_mmseg-ssn-vitt.py"
    elif args.model == "mmseg-ssn-vitp":
        entry_script = "run_mmseg-ssn-vitp.py"
    elif args.model == "mmseg-ssn-vit-hi-mha":
        entry_script = "run_mmseg-ssn-hi-mha.py"
    elif args.model == "mmseg-ssn-vit-hi-mha_4_8":
        entry_script = "run_mmseg-ssn-hi-mha_4_8.py"
    elif args.model == "mmseg-ssn-vit-hi-mha_4_8_10":
        entry_script = "run_mmseg-ssn-hi-mha_4_8_10.py"
    elif args.model == "mmseg-ssn-vit-hi-mha_4_10":
        entry_script = "run_mmseg-ssn-hi-mha_4_10.py"
    elif args.model == "mmseg-ssn-vit-hi-mha_unrefine":
        entry_script = "run_mmseg-ssn-hi-mha_unrefine.py"
    elif args.model == "mmseg-ssn-vit-hi-mha_4_8_unrefine":
        entry_script = "run_mmseg-ssn-hi-mha_4_8_unrefine.py"
    elif args.model == "mmseg-ssn-vit-hi-mha_4_8_10_unrefine":
        entry_script = "run_mmseg-ssn-hi-mha_4_8_10_unrefine.py"
    elif args.model == "mmseg-ssn-vit-hi-mha_4_10_unrefine":
        entry_script = "run_mmseg-ssn-hi-mha_4_10_unrefine.py"
    elif args.model == "mmseg-ssn-vit-hi-mha-p":
        entry_script = "run_mmseg-ssn-hi-mha-p.py"
    elif args.model == "mmseg-ssn-vit-hi-mha-p_4_8":
        entry_script = "run_mmseg-ssn-hi-mha-p_4_8.py"
    elif args.model == "mmseg-ssn-vit-hi-mha-p_4_8_10":
        entry_script = "run_mmseg-ssn-hi-mha-p_4_8_10.py"
    elif args.model == "mmseg-ssn-vit-hi-mha-p_4_10":
        entry_script = "run_mmseg-ssn-hi-mha-p_4_10.py"
    elif args.model == "mmseg-ssn-vit-hi-mha-p_unrefine":
        entry_script = "run_mmseg-ssn-hi-mha-p_unrefine.py"
    elif args.model == "mmseg-ssn-vit-hi-mha-p_4_8_unrefine":
        entry_script = "run_mmseg-ssn-hi-mha-p_4_8_unrefine.py"
    elif args.model == "mmseg-ssn-vit-hi-mha-p_4_8_10_unrefine":
        entry_script = "run_mmseg-ssn-hi-mha-p_4_8_10_unrefine.py"
    elif args.model == "mmseg-ssn-vit-hi-mha-p_4_10_unrefine":
        entry_script = "run_mmseg-ssn-hi-mha-p_4_10_unrefine.py"

    elif args.model == "mmseg-ssn-vit-u_2_4_8_10":
        entry_script = "run_mmseg-ssn-u_2_4_8_10.py"
    elif args.model == "mmseg-ssn-vit-u_3_5_8_11":
        entry_script = "run_mmseg-ssn-u_3_5_8_11.py"
    elif args.model == "mmseg-ssn-vit-u_5_10":
        entry_script = "run_mmseg-ssn-u_5_10.py"
    elif args.model == "mmseg-ssn-vit-u_4_9":
        entry_script = "run_mmseg-ssn-u_4_9.py"

    elif args.model == "mmseg-ssn-vit-u_2_4_8_10_unrefine":
        entry_script = "run_mmseg-ssn-u_2_4_8_10_unrefine.py"
    elif args.model == "mmseg-ssn-vit-u_3_5_8_11_unrefine":
        entry_script = "run_mmseg-ssn-u_3_5_8_11_unrefine.py"
    elif args.model == "mmseg-ssn-vit-u_5_10_unrefine":
        entry_script = "run_mmseg-ssn-u_5_10_unrefine.py"
    elif args.model == "mmseg-ssn-vit-u_4_9_unrefine":
        entry_script = "run_mmseg-ssn-u_4_9_unrefine.py"

    elif args.model == "mmseg-ssn-vit-u_2_4_8_10_s":
        entry_script = "run_mmseg-ssn-u_2_4_8_10_s.py"
    elif args.model == "mmseg-ssn-vit-u_3_5_8_11_s":
        entry_script = "run_mmseg-ssn-u_3_5_8_11_s.py"
    elif args.model == "mmseg-ssn-vit-u_5_10_s":
        entry_script = "run_mmseg-ssn-u_5_10_s.py"
    elif args.model == "mmseg-ssn-vit-u_4_9_s":
        entry_script = "run_mmseg-ssn-u_4_9_s.py"
    
  
    elif args.model == "mmseg-ssn-vits":
        entry_script = "run_mmseg-ssn-vits.py"
    elif args.model == "mlm_ade_training":
        entry_script = "run_mlm_ade_training.py"
    elif args.model == "beit_mask_fin":
        entry_script = "run_beit_mask_fin.py"
    elif args.model == "beit_mask_fine":
        entry_script = "run_beit_mask_fine.py"
    elif args.model == "mmseg-ssn-vit-p-hi-mha":
        entry_script = "run_mmseg-ssn-vit-p-hi-mha.py"
    elif args.model == "mmseg-ssn-vit-p-rhi-mha":
        entry_script = "run_mmseg-ssn-rhi-mha_4_10.py"
    elif args.model == "mmseg-ssn-vit-p-rdhi-mha":
        entry_script = "run_mmseg-ssn-rdhi-mha_4_10.py"
    elif args.model == "mmseg-ssn-vit-p-rdfhi-mha":
        entry_script = "run_mmseg-ssn-rdfhi-mha.py"
    elif args.model == "mmseg-ssn-vit-p-rdfhi-mha_o":
        entry_script = "run_mmseg-ssn-rdfhi-mha_o.py" 
        
    elif args.model == "mmseg-ssn-vit-p-rdfhi-mha_1":
        entry_script = "run_mmseg-ssn-rdfhi-mha_1.py"
    elif args.model == "mmseg-ssn-vit-p-rdfhi-mha_o_1":
        entry_script = "run_mmseg-ssn-rdfhi-mha_o_1.py" 
    elif args.model == "mmseg-ssn-vit-p-rdfhi-mha_o_2":
        entry_script = "run_mmseg-ssn-rdfhi-mha_o_2.py" 
    elif args.model == "mmseg-ssn-vit-p-rdfhi-mha_o_3":
        entry_script = "run_mmseg-ssn-rdfhi-mha_o_3.py" 
    elif args.model == "mmseg-ssn-vit-p-rdfhi-mha_o_4":
        entry_script = "run_mmseg-ssn-rdfhi-mha_o_4.py" 

    elif args.model == "mmseg-vit_ori":
        entry_script = "run_mmseg_seg_vit_ori.py"
    elif args.model == "mmseg-vit_ori_ade20k":
        entry_script = "run_mmseg_seg_vit_ori_ade20k.py"
    elif args.model == "mmseg-vit_ssn":
        entry_script = "run_mmseg_seg_vit_ssn.py"
    elif args.model == "mmseg-vit_ssn_ade20k":
        entry_script = "run_mmseg_seg_vit_ssn_ade20k.py"
    elif args.model == "mmseg-vit_ssn_L":
        entry_script = "run_mmseg_seg_vit_ssn_L.py"
    elif args.model == "mmseg-vit_ssn_S":
        entry_script = "run_mmseg_seg_vit_ssn_S.py"
    elif args.model == "mmseg-vit_ssn_SC":
        entry_script = "run_mmseg_seg_vit_ssn_SC.py"
    elif args.model == "mmseg-vit_ssn_SC_ade20k":
        entry_script = "run_mmseg_seg_vit_ssn_SC_ade20k.py"
    elif args.model == "mmseg-vit_ssn_S_topk_ade20k":
        entry_script = "run_mmseg_seg_vit_ssn_S_topk_ade20k.py"
    elif args.model == "mmseg-vit_ssn_S_topk_offset":
        entry_script = "run_mmseg_seg_vit_ssn_S_topk_offset.py"
    elif args.model == "mmseg-vit_ssn_S_topk_offset_ade20k":
        entry_script = "run_mmseg_seg_vit_ssn_S_topk_offset_ade20k.py"
   



    elif args.model == "mmseg-vit_ssn_S_topk":
        entry_script = "run_mmseg_seg_vit_ssn_S_topk.py"
    elif args.model == "mmseg-vit_ssn_10_S_topk":
        entry_script = "run_mmseg_seg_vit_ssn_10_S_topk.py"
    elif args.model == "mmseg-vit_ssn_10_S_topk_ade20k":
        entry_script = "run_mmseg_seg_vit_ssn_10_S_topk_ade20k.py"

    elif args.model == "mmseg-vit_ssn_6_7_S_topk":
        entry_script = "run_mmseg_seg_vit_ssn_6_7_S_topk.py"
    elif args.model == "mmseg-vit_ssn_6_7_S_topk_ade20k":
        entry_script = "run_mmseg_seg_vit_ssn_6_7_S_topk_ade20k.py"

    elif args.model == "mmseg-vit_ssn_8_9_S_topk":
        entry_script = "run_mmseg_seg_vit_ssn_8_9_S_topk.py"
    elif args.model == "mmseg-vit_ssn_8_9_S_topk_ade20k":
        entry_script = "run_mmseg_seg_vit_ssn_8_9_S_topk_ade20k.py"

    elif args.model == "mmseg-vit_ssn_8_11_S_topk":
        entry_script = "run_mmseg_seg_vit_ssn_8_11_S_topk.py"
    elif args.model == "mmseg-vit_ssn_8_11_S_topk_ade20k":
        entry_script = "run_mmseg_seg_vit_ssn_8_11_S_topk_ade20k.py"
    elif args.model == "mmseg-vit_ssn_9_11_S_topk":
        entry_script = "run_mmseg_seg_vit_ssn_9_11_S_topk.py"
    elif args.model == "mmseg-vit_ssn_9_11_S_topk_ade20k":
        entry_script = "run_mmseg_seg_vit_ssn_9_11_S_topk_ade20k.py"
    elif args.model == "mmseg-vit_ssn_6_8_S_topk":
        entry_script = "run_mmseg_seg_vit_ssn_6_8_S_topk.py"
    elif args.model == "mmseg-vit_ssn_6_8_S_topk_ade20k":
        entry_script = "run_mmseg_seg_vit_ssn_6_8_S_topk_ade20k.py"

    
    elif args.model == "mmseg-vit_ssn_S_topk_M":
        entry_script = "run_mmseg_seg_vit_ssn_S_topk_M.py"
    elif args.model == "mmseg-vit_ssn_S_topk_M_ade20k":
        entry_script = "run_mmseg_seg_vit_ssn_S_topk_M_ade20k.py"
    elif args.model == "mmseg-vit_ada_D":
        entry_script = "run_mmseg_ada_D.py"
    elif args.model == "mmseg-vit_ada_HD":
        entry_script = "run_mmseg_ada_HD.py"
    elif args.model == "mmseg-vit_Q":
        entry_script = "run_mmseg_seg_vit_ssn_Q.py"
    elif args.model == "mmseg-vit_Q_ade20k":
        entry_script = "run_mmseg_seg_vit_ssn_Q_ade20k.py"

    elif args.model == "mmseg-vit_ssn_H_topk_ade20k":
        entry_script = "run_mmseg_seg_vit_ssn_H_topk_ade20k.py"
    elif args.model == "mmseg-vit_ssn_H_topk":
        entry_script = "run_mmseg_seg_vit_ssn_H_topk.py"
    elif args.model == "mmseg-vit_ssn_H_ade20k":
        entry_script = "run_mmseg_seg_vit_ssn_H_ade20k.py"
    elif args.model == "mmseg-vit_ssn_H":
        entry_script = "run_mmseg_seg_vit_ssn_H.py"

    elif args.model == "mmseg-vit_ssn_C_topk_ade20k":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk_ade20k.py"
    elif args.model == "mmseg-vit_ssn_C_topk":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk.py"

    elif args.model == "mmseg-vit_ssn_CMS_topk_ade20k":
        entry_script = "run_mmseg_seg_vit_ssn_CMS_topk_ade20k.py"
    elif args.model == "mmseg-vit_ssn_CMS_topk":
        entry_script = "run_mmseg_seg_vit_ssn_CMS_topk.py"

    elif args.model == "mmseg-vit_ssn_C_topk_ade20k_2":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk_ade20k_2.py"
    elif args.model == "mmseg-vit_ssn_C_topk_2":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk_2.py"

    elif args.model == "mmseg-vit_ssn_C_topk_ade20k_3":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk_ade20k_3.py"
    elif args.model == "mmseg-vit_ssn_C_topk_3":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk_3.py"

    elif args.model == "mmseg-vit_ssn_C_topk_ade20k_4":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk_ade20k_4.py"
    elif args.model == "mmseg-vit_ssn_C_topk_4":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk_4.py"

    elif args.model == "mmseg-vit_ssn_C_topk_ade20k_5":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk_ade20k_5.py"
    elif args.model == "mmseg-vit_ssn_C_topk_5":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk_5.py"

    elif args.model == "mmseg-vit_ssn_C_topk_ade20k_6":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk_ade20k_6.py"
    elif args.model == "mmseg-vit_ssn_C_topk_6":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk_6.py"

    elif args.model == "mmseg-vit_ssn_C_topk_ade20k_7":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk_ade20k_7.py"
    elif args.model == "mmseg-vit_ssn_C_topk_7":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk_7.py"

    elif args.model == "mmseg-vit_ssn_C_topk_ade20k_8":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk_ade20k_8.py"
    elif args.model == "mmseg-vit_ssn_C_topk_8":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk_8.py"

    elif args.model == "mmseg-vit_ssn_C_topk_ade20k_9":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk_ade20k_9.py"
    elif args.model == "mmseg-vit_ssn_C_topk_9":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk_9.py"

    elif args.model == "mmseg-vit_ssn_C_topk_ade20k_10":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk_ade20k_10.py"
    elif args.model == "mmseg-vit_ssn_C_topk_10":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk_10.py"

    elif args.model == "mmseg-vit_ssn_C_topk_ade20k_11":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk_ade20k_11.py"
    elif args.model == "mmseg-vit_ssn_C_topk_11":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk_11.py"

    elif args.model == "mmseg-vit_ssn_C_topk_ade20k_12":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk_ade20k_12.py"
    elif args.model == "mmseg-vit_ssn_C_topk_12":
        entry_script = "run_mmseg_seg_vit_ssn_C_topk_12.py"

    elif "mmseg-vit_ssn" in args.model:
        txt = args.model.split("-")
        entry_script = "run_mmseg_seg_"+txt[-1]+".py"

    


    
    
    else:
        # model: hrnet ocr
        entry_script = "run_{}_{}.py".format(args.model, dataset)

    def make_container_registry(address):
        cr = ContainerRegistry()
        cr.address = address
        return cr

    my_registry = make_container_registry(address=container_registry_address)
    mpi_config = MpiConfiguration()
    mpi_config.process_count_per_node = 1
    nccl = Nccl()

    if args.nnode >= 2:
        estimator = PyTorch(
            source_directory="./docker",
            script_params=script_params,
            compute_target=ct,
            use_gpu=True,
            node_count=args.nnode,
            distributed_training=mpi_config,
            shm_size="400G",
            image_registry_details=my_registry,
            entry_script=entry_script,
            custom_docker_image=custom_docker_image,
            user_managed=True,
        )
    else:
        estimator = PyTorch(
            source_directory="./docker",
            script_params=script_params,
            compute_target=ct,
            use_gpu=True,
            shm_size="400G",
            image_registry_details=my_registry,
            entry_script=entry_script,
            custom_docker_image=custom_docker_image,
            user_managed=True,
        )
    cmk8sconfig = K8sComputeConfiguration()

    cmk8s = dict()
    cmk8s["gpu_count"] = args.num_gpus
    cmk8s["enable_ipython"] = True

    if args.preemp:
        cmk8s["preemption_allowed"] = True

    cmk8sconfig.configuration = cmk8s
    estimator.run_config.cmk8scompute = cmk8sconfig

    experiment = Experiment(ws, name=config.experiment_name)
    run = experiment.submit(estimator, tags={"tag": args.cfg})

    pprint.pprint(run)
