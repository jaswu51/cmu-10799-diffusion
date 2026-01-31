
-------------------Q2
source /Users/yiwu/Documents/Github/modal/cmu-10799-diffusion/.venv-cpu/bin/activate

modal run modal_app.py --action train --method ddpm --config configs/flow_matching.yaml

bash scripts/evaluate_modal_torch_fidelity.sh \
  --method flow_matching \
  --checkpoint /logs/flow_matching/flow_matching_20260127_182922/checkpoints/flow_matching_final.pt \
  --metrics kid \
  --num-samples 1000 \
  --num-steps 50

kernel_inception_distance_mean: 0.005613346
kernel_inception_distance_std: 0.0003327286

-------------------Q4
modal run modal_app.py --action sample \
  --checkpoint logs/ddpm_modal/ddpm_20260121_213636/checkpoints/ddpm_final.pt \
  --method ddpm \
  --sampler ddim \
  --eta 0.0 \
  --num-samples 16 \
  --num-steps 100 

Samples saved to /data/samples/ddpm_20260128_021939.png
modal volume get cmu-10799-diffusion-data "/samples/ddpm_20260128_021939.png" "hw_answers/hw2/q4/ddpm_20260128_021939.png"


bash scripts/evaluate_modal_torch_fidelity.sh \
  --method ddpm \
  --sampler ddim \
  --eta 0.0 \
  --checkpoint logs/ddpm_modal/ddpm_20260121_213636/checkpoints/ddpm_final.pt \
  --metrics kid \
  --num-samples 1000 \
  --num-steps 100

kernel_inception_distance_mean: 0.004699779
kernel_inception_distance_std: 0.0003207737

