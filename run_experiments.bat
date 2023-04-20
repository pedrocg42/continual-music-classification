@REM Joint training

python execute_experiment.py --experiment gtzan_resnet50_joint
python execute_experiment.py --experiment gtzan_resnet50dino_joint

python execute_experiment.py --experiment gtzan_frozenresnet50_joint
python execute_experiment.py --experiment gtzan_frozenresnet50dino_joint

python execute_experiment.py --experiment gtzan_vqresnet50_joint
python execute_experiment.py --experiment gtzan_vqresnet50dino_joint

python execute_experiment.py --experiment gtzan_dkvbresnet50_joint
python execute_experiment.py --experiment gtzan_dkvbresnet50dino_joint

@REM Scenario 1

python execute_experiment.py --experiment gtzan_resnet50_scenario1
python execute_experiment.py --experiment gtzan_resnet50dino_scenario1

python execute_experiment.py --experiment gtzan_frozenresnet50_scenario1
python execute_experiment.py --experiment gtzan_frozenresnet50dino_scenario1

python execute_experiment.py --experiment gtzan_vqresnet50_scenario1
python execute_experiment.py --experiment gtzan_vqresnet50dino_scenario1

python execute_experiment.py --experiment gtzan_dkvbresnet50_scenario1
python execute_experiment.py --experiment gtzan_dkvbresnet50dino_scenario1