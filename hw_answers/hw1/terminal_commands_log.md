
-------------------Q4-RUN1
source /Users/yiwu/Documents/Github/modal/cmu-10799-diffusion/.venv-cpu/bin/activate

modal run modal_app.py --action train --method ddpm --config configs/ddpm_modal_eps.yaml

modal run modal_app.py --action sample \
  --method ddpm \
  --checkpoint logs/ddpm_modal/ddpm_20260121_213636/checkpoints/ddpm_final.pt \
  --num-samples 16 \
  --num-steps 1000

bash scripts/evaluate_modal_torch_fidelity.sh \
  --method ddpm \
  --checkpoint logs/ddpm_modal/ddpm_20260121_213636/checkpoints/ddpm_final.pt \
  --metrics kid \
  --num-samples 1000 \
  --num-steps 1000

kernel_inception_distance_mean: 0.004591246, 
kernel_inception_distance_std: 0.0003308931

-------------------Q4-RUN2
source /Users/yiwu/Documents/Github/modal/cmu-10799-diffusion/.venv-cpu/bin/activate

modal run modal_app.py --action train --method ddpm --config configs/ddpm_modal_eps.yaml

modal run modal_app.py --action sample \
  --method ddpm \
  --checkpoint logs/ddpm_modal_eps/ddpm_20260124_051327/checkpoints/ddpm_final.pt \
  --num-samples 16 \
  --num-steps 1000

modal volume get cmu-10799-diffusion-data "/samples/ddpm_20260124_190801.png" "hw_answers/hw1/q4/ddpm_20260124_190801.png"

bash scripts/evaluate_modal_torch_fidelity.sh \
  --method ddpm \
  --checkpoint logs/ddpm_modal_eps/ddpm_20260124_051327/checkpoints/ddpm_final.pt \
  --metrics kid \
  --num-samples 1000 \
  --num-steps 1000


kernel_inception_distance_mean: 0.006668191
kernel_inception_distance_std: 0.0003726493

-------------------Q6
source /Users/yiwu/Documents/Github/modal/cmu-10799-diffusion/.venv-cpu/bin/activate

modal run modal_app.py --action train --method ddpm --config configs/ddpm_modal_v.yaml

modal run modal_app.py --action sample \
  --method ddpm \
  --checkpoint logs/ddpm_modal_v/ddpm_20260123_205410/checkpoints/ddpm_final.pt \
  --num-samples 16 \
  --num-steps 1000

modal volume get cmu-10799-diffusion-data "/samples/ddpm_20260124_044516.png" "hw_answers/hw1/q6/ddpm_20260124_044516.png"

bash scripts/evaluate_modal_torch_fidelity.sh \
  --method ddpm \
  --checkpoint logs/ddpm_modal_v/ddpm_20260123_205410/checkpoints/ddpm_final.pt \
  --metrics kid \
  --num-samples 1000 \
  --num-steps 1000

kernel_inception_distance_mean: 0.005799966
kernel_inception_distance_std: 0.0003250741

-------------------Q7
# 100 steps
bash scripts/evaluate_modal_torch_fidelity.sh --num-steps 100 --num-samples 1000 --checkpoint logs/ddpm_modal/ddpm_20260121_213636/checkpoints/ddpm_final.pt --override

kernel_inception_distance_mean: 0.6353165
kernel_inception_distance_std: 0.002024705

# 300 steps
bash scripts/evaluate_modal_torch_fidelity.sh --num-steps 300 --num-samples 1000 --checkpoint logs/ddpm_modal/ddpm_20260121_213636/checkpoints/ddpm_final.pt --override

kernel_inception_distance_mean: 0.6057082
kernel_inception_distance_std: 0.00191208


# 500 steps
bash scripts/evaluate_modal_torch_fidelity.sh --num-steps 500 --num-samples 1000 --checkpoint logs/ddpm_modal/ddpm_20260121_213636/checkpoints/ddpm_final.pt --override

kernel_inception_distance_mean: 0.4693184
kernel_inception_distance_std: 0.001870076

# 700 steps
bash scripts/evaluate_modal_torch_fidelity.sh --num-steps 700 --num-samples 1000 --checkpoint logs/ddpm_modal/ddpm_20260121_213636/checkpoints/ddpm_final.pt --override

kernel_inception_distance_mean: 0.1775895
kernel_inception_distance_std: 0.001563413

# 900 steps
bash scripts/evaluate_modal_torch_fidelity.sh --num-steps 900 --num-samples 1000 --checkpoint logs/ddpm_modal/ddpm_20260121_213636/checkpoints/ddpm_final.pt --override

kernel_inception_distance_mean: 0.008109298
kernel_inception_distance_std: 0.0004136323


# 1000 steps (Baseline)
bash scripts/evaluate_modal_torch_fidelity.sh --num-steps 1000 --num-samples 1000 --checkpoint logs/ddpm_modal/ddpm_20260121_213636/checkpoints/ddpm_final.pt --override

kernel_inception_distance_mean: 0.004699788
kernel_inception_distance_std: 0.0003207569


for steps in 100 300 500 700 900 1000; do
  modal run modal_app.py --action sample --num-samples 1 --num-steps $steps \
    --checkpoint logs/ddpm_modal/ddpm_20260121_213636/checkpoints/ddpm_final.pt
done