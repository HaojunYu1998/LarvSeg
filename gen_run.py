
f = open("run_eval.sh", "w")
for i, out_dir in enumerate([
        # "work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_ade150_prior_loss",
        # "work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_ade150_structure_loss",
        # "work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_ade847_prior_loss",
        # "work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_ade847_prior_structure_loss",
        # "work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_ade847_structure_loss",
        # "work_dirs/20221006_large_voc_vitb16_linear_80k_bs16_ade847_prior_loss",
        # "work_dirs/20221006_large_voc_vitb16_linear_80k_bs16_ade847_prior_structure_loss",
        # "work_dirs/20221006_large_voc_vitb16_linear_80k_bs16_cocostuff_prior_loss",
        # "work_dirs/20221006_large_voc_vitb16_linear_80k_bs16_cocostuff_prior_structure_loss"
        "work_dirs/20221006_large_voc_vitb16_linear_80k_bs16_ade150_prior_loss",
        "work_dirs/20221006_large_voc_vitb16_linear_80k_bs16_ade150_prior_structure_loss"
    ]):
    for data in ["coco171", "ade150", "ade847"]:
        for o in [1,3,10]:
            if "cosine" in out_dir:
                model = "cosine"
            else:
                model = "linear"
            
            cfg = f"configs/large_voc/vit/oracle/large_voc_vitb16_{model}_eval_{data}_oracle{o}.py"

            if "prior_structure" in out_dir:
                mode = "prior_structure"
            elif "prior" in out_dir:
                mode = "prior"
            else:
                mode = "structure"
            for v in range(5):
                print(f"bash tools/dist_test.sh {cfg} {out_dir}/iter_80000.pth 4 --eval mIoU > {i+9}{data}_{mode}_point{o}_v{v}.txt", file=f)
f.close()
