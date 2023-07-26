SET DATASETS_PATH=G:/Pedro/data/music
@REM Oracle training

python execute_experiment_check.py --experiment mert95m_base_oracle_gtzan_all

@REM Scenario 1

python execute_experiment_check.py --experiment mert95m_finetuning_cl_gtzan_scenario1
python execute_experiment_check.py --experiment mert95m_replay_cl_gtzan_scenario1
python execute_experiment_check.py --experiment mert95m_dkvb_cl_gtzan_scenario1
python execute_experiment_check.py --experiment mert95m_gem_cl_gtzan_scenario1
python execute_experiment_check.py --experiment mert95m_ewc_cl_gtzan_scenario1
python execute_experiment_check.py --experiment mert95m_l2p_cl_gtzan_scenario1


@REM Turn off VM (comment if not needed)
@REM sudo shutdown