
cd /content/drive/MyDrive/voice-biometrics-detection
python -c "from vbd import bootstrap; bootstrap.main(with_pool_cache=False)"
python -m vbd.train_cm --model lcnn --epochs 30 --bs 64 --lr 3e-4 --resume
echo LCNN_DONE
python -m vbd.train_cm --model rawnet2 --epochs 20 --bs 32 --lr 1e-4 --resume
echo RAWNET2_DONE
