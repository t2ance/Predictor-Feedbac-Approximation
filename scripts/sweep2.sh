#ps aux | grep "main_sweep"
#pkill -f "main_sweep"
#pkill -f "LSTM-DeepONet"
python main_sweep.py -s "s9" -model_name "LSTM-FNO" -device 'cuda:5' &