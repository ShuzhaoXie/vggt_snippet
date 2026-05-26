```
python run_all_test.py --model vggt_omega --device 7 --checkpoint /mnt/storage2/users/szxie_data/vggt_omega/vggt_omega_1b_512.pt --output_dir /mnt/storage2/users/szxie_data/agentrecon/geo_priors

python run_all_test.py --model vggt_omega --device 0 --checkpoint /autodl-fs/data/ckpts/vggt_omega_1b_512.pt --output_dir /autodl-fs/data/geo_priors_v2 --data-root /autodl-fs/data --max-frames -1

python run_all_test.py --model vggt --device 0 --checkpoint /autodl-fs/data/ckpts/model_vggt.pt --output_dir /autodl-fs/data/geo_priors_v2 --data-root /autodl-fs/data --max-frames -1

python run_vggt.py \
  --input-path /autodl-fs/data/ras/hallway.mp4 \
  --dataset-name ras \
  --scene-name hallway \
  --output-dir /autodl-fs/data/geo_priors_v2 \
  --model vggt_omega \
  --checkpoint /autodl-fs/data/ckpts/vggt_omega_1b_512.pt \
  --device 0 \
  --max-frames -1