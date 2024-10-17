#ps aux | grep "main_sweep" | grep "peijia"
#pkill -f "DeepONet"
python main_sweep.py -s "s9" -model_name "LSTM-DeepONet" -device 'cuda:6' &
python main_sweep.py -s "s9" -model_name "GRU-DeepONet" -device 'cuda:6' &
python main_sweep.py -s "s9" -model_name "DeepONet-GRU" -device 'cuda:7' &
python main_sweep.py -s "s9" -model_name "DeepONet-LSTM" -device 'cuda:7' &
python main_sweep.py -s "s9" -model_name "DeepONet" -device 'cuda:5' &
#python main_sweep.py -s "s11" -model_name "LSTM-DeepONet" -device 'cuda:7' &
#python main_sweep.py -s "s11" -model_name "GRU-DeepONet" -device 'cuda:7' &