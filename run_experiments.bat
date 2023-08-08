SET DATASETS_PATH=G:/Pedro/data/music
@REM Oracle training

@REM python execute_experiment.py --experiment mert95m_base_oracle_gtzan_all

@REM GTZAN

@REM python execute_experiment.py --experiment mert95m_finetuning_cl_gtzan_scenario1
@REM python execute_experiment.py --experiment mert95m_replay_cl_gtzan_scenario1
python execute_experiment.py --experiment mert95m_icarl_cl_gtzan_scenario1
@REM python execute_experiment.py --experiment mert95m_dkvb_cl_gtzan_scenario1
@REM python execute_experiment.py --experiment mert95m_gem_cl_gtzan_scenario1
@REM python execute_experiment.py --experiment mert95m_ewc_cl_gtzan_scenario1
python execute_experiment.py --experiment mert95m_l2p_cl_gtzan_scenario1
python execute_experiment.py --experiment mert95m_embcenter_cl_gtzan_scenario1


@REM VocalSet Singer

python execute_experiment.py --experiment mert95m_finetuning_cl_vocalsetsinger_scenario1
python execute_experiment.py --experiment mert95m_replay_cl_vocalsetsinger_scenario1
python execute_experiment.py --experiment mert95m_icarl_cl_vocalsetsinger_scenario1
python execute_experiment.py --experiment mert95m_dkvb_cl_vocalsetsinger_scenario1
python execute_experiment.py --experiment mert95m_gem_cl_vocalsetsinger_scenario1
python execute_experiment.py --experiment mert95m_ewc_cl_vocalsetsinger_scenario1
python execute_experiment.py --experiment mert95m_l2p_cl_vocalsetsinger_scenario1
python execute_experiment.py --experiment mert95m_embcenter_cl_vocalsetsinger_scenario1


@REM VocalSet Tech

python execute_experiment.py --experiment mert95m_finetuning_cl_vocalsettech_scenario1
python execute_experiment.py --experiment mert95m_replay_cl_vocalsettech_scenario1
python execute_experiment.py --experiment mert95m_icarl_cl_vocalsettech_scenario1
python execute_experiment.py --experiment mert95m_dkvb_cl_vocalsettech_scenario1
python execute_experiment.py --experiment mert95m_gem_cl_vocalsettech_scenario1
python execute_experiment.py --experiment mert95m_ewc_cl_vocalsettech_scenario1
python execute_experiment.py --experiment mert95m_l2p_cl_vocalsettech_scenario1
python execute_experiment.py --experiment mert95m_embcenter_cl_vocalsettech_scenario1


@REM NSynth Instrument

python execute_experiment.py --experiment mert95m_finetuning_cl_nsynthinstrument_scenario1
python execute_experiment.py --experiment mert95m_replay_cl_nsynthinstrument_scenario1
python execute_experiment.py --experiment mert95m_icarl_cl_nsynthinstrument_scenario1
python execute_experiment.py --experiment mert95m_dkvb_cl_nsynthinstrument_scenario1
python execute_experiment.py --experiment mert95m_gem_cl_nsynthinstrument_scenario1
python execute_experiment.py --experiment mert95m_ewc_cl_nsynthinstrument_scenario1
python execute_experiment.py --experiment mert95m_l2p_cl_nsynthinstrument_scenario1
python execute_experiment.py --experiment mert95m_embcenter_cl_nsynthinstrument_scenario1


@REM Turn off VM (comment if not needed)
@REM sudo shutdown