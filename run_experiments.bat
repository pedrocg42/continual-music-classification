@REM Joint training

python execute_experiment.py --experiment Mert95m_joint_gtzan_all
python execute_experiment.py --experiment Mert95mVq_joint_gtzan_all
python execute_experiment.py --experiment Mert95mDkvb_joint_gtzan_all

@REM Scenario 1

python execute_experiment.py --experiment Mert95m_cl_gtzan_scenario1
python execute_experiment.py --experiment Mert95mVq_cl_gtzan_scenario1
python execute_experiment.py --experiment Mert95mDkvb_cl_gtzan_scenario1