#ps aux | grep "main_sweep" | grep "peijia"
#pkill -f "DeepONet"
python main_sweep.py -s "s11" -model_name "FNO" -device 'cuda:4' -z2u true &
python main_sweep.py -s "s11" -model_name "DeepONet" -device 'cuda:4' -z2u true &
python main_sweep.py -s "s11" -model_name "GRU" -device 'cuda:4' -z2u true &
python main_sweep.py -s "s11" -model_name "LSTM" -device 'cuda:1' -z2u true &
python main_sweep.py -s "s11" -model_name "FNO-LSTM" -device 'cuda:1' -z2u true &
python main_sweep.py -s "s11" -model_name "FNO-GRU" -device 'cuda:1' -z2u true &
python main_sweep.py -s "s11" -model_name "GRU-FNO" -device 'cuda:2' -z2u true &
python main_sweep.py -s "s11" -model_name "LSTM-FNO" -device 'cuda:2' -z2u true &
python main_sweep.py -s "s11" -model_name "DeepONet-LSTM" -device 'cuda:2' -z2u true &
python main_sweep.py -s "s11" -model_name "DeepONet-GRU" -device 'cuda:3' -z2u true &
python main_sweep.py -s "s11" -model_name "GRU-DeepONet" -device 'cuda:3' -z2u true &
python main_sweep.py -s "s11" -model_name "LSTM-DeepONet" -device 'cuda:3' -z2u true &