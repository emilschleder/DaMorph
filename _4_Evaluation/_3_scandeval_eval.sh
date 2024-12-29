# To run this using ITU HPC it requires that you have created a conda env using the requirements from _4_Evaluation/scandeval/hpc_scandeval_env.yaml
python _4_Evaluation/scandeval/_0_setup.py \
--package_path /home/easc/.conda/envs/scandeval_env_3/lib/python3.11/site-packages/scandeval/ \
--local_path _4_Evaluation/scandeval/files \
&& python _4_Evaluation/scandeval/_1_submit_scandeval.py \
--models_file _4_Evaluation/scandeval/eval_models.txt \
--hpc_env scandeval_env_3