#  Joint training

python execute_experiment.py --experiment mert95m_base_joint_gtzan_all
python execute_experiment.py --experiment mert95m_vq_joint_gtzan_all
python execute_experiment.py --experiment mert95m_dkvb_joint_gtzan_all

#  Scenario 1

python execute_experiment.py --experiment mert95m_base_cl_gtzan_scenario1
python execute_experiment.py --experiment mert95m_vq_cl_gtzan_scenario1
python execute_experiment.py --experiment mert95m_dkvb_cl_gtzan_scenario1

# Turn off VM (comment if not needed)
sudo shutdown