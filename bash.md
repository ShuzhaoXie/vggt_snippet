```
python run_all_test.py --model vggt_omega --device 7 --checkpoint /mnt/storage2/users/szxie_data/vggt_omega/vggt_omega_1b_512.pt --output_dir /mnt/storage2/users/szxie_data/agentrecon/geo_priors

python run_all_test.py --model vggt_omega --device 0 --checkpoint /autodl-fs/data/ckpts/vggt_omega_1b_512.pt --output_dir /autodl-fs/data/geo_priors --data-root /autodl-fs/data --max-frames -1