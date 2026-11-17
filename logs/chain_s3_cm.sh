
cd /content/drive/MyDrive/voice-biometrics-detection
while ! grep -q ADAPT_READY logs/prep_adapt.log; do sleep 15; done
for k in 0 1 2 3 4; do python -m vbd.train_sv3 --base stage2v2 --name stage3 --rot $k; done
echo STAGE3_ALL_DONE
python -m vbd.train_cm --model lcnn --epochs 30 --bs 64 --lr 3e-4 --resume
echo LCNN_DONE
python -m vbd.train_cm --model rawnet2 --epochs 20 --bs 32 --lr 1e-4 --resume
echo RAWNET2_DONE
