python -m hparams_search_scripts.run_mechanism_hparam_search --config configs_final/facebook_trial/figure0_clean.yaml --output_root_dir paper_experiments/facebook_test/clean
python -m hparams_search_scripts.run_mechanism_hparam_search --config configs_final/facebook_trial/random_projected.yaml --output_root_dir paper_experiments/facebook_test/random
python -m hparams_search_scripts.run_mechanism_hparam_search --config configs_final/facebook_trial/learned_projected.yaml --output_root_dir paper_experiments/facebook_test/adapt
