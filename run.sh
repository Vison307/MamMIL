GPUS=0,

for fold in {0..9}
do
    /usr/local/bin/python3 train.py --stage train --config configs/BRCAS/MamMIL2.yaml --gpus $GPUS --fold $fold --seed 1
    /usr/local/bin/python3 train.py --stage test --config configs/BRCAS/MamMIL2.yaml --gpus $GPUS --fold $fold --seed 1
done

for fold in {0..9}
do
    /usr/local/bin/python3 train.py --stage train --config configs/BRCAS-VIM/MamMIL2.yaml --gpus $GPUS --fold $fold --seed 1
    /usr/local/bin/python3 train.py --stage test --config configs/BRCAS-VIM/MamMIL2.yaml --gpus $GPUS --fold $fold --seed 1
done


for fold in {0..9}
do
    /usr/local/bin/python3 train.py --stage train --config configs/BRCAS-VMAMBA/MamMIL2.yaml --gpus $GPUS --fold $fold --seed 1
    /usr/local/bin/python3 train.py --stage test --config configs/BRCAS-VMAMBA/MamMIL2.yaml --gpus $GPUS --fold $fold --seed 1
done

/usr/local/bin/python3 train.py --stage train --config configs/BRCAS-official/MamMIL2.yaml --gpus $GPUS --fold 0 --seed 1
/usr/local/bin/python3 train.py --stage test --config configs/BRCAS-official/MamMIL2.yaml --gpus $GPUS --fold 0 --seed 1

/usr/local/bin/python3 train.py --stage train --config configs/BRCAS-official-VIM/MamMIL2.yaml --gpus $GPUS --fold 0 --seed 1
/usr/local/bin/python3 train.py --stage test --config configs/BRCAS-official-VIM/MamMIL2.yaml --gpus $GPUS --fold 0 --seed 1

/usr/local/bin/python3 train.py --stage train --config configs/BRCAS-official-VMAMBA/MamMIL2.yaml --gpus $GPUS --fold 0 --seed 1
/usr/local/bin/python3 train.py --stage test --config configs/BRCAS-official-VMAMBA/MamMIL2.yaml --gpus $GPUS --fold 0 --seed 1

/usr/local/bin/python3 train.py --stage train --config configs/Camelyon16-transmil/MamMIL2.yaml --gpus $GPUS --fold 0 --seed 1
/usr/local/bin/python3 train.py --stage test --config configs/Camelyon16-transmil/MamMIL2.yaml --gpus $GPUS --fold 0 --seed 1

/usr/local/bin/python3 train.py --stage train --config configs/Camelyon16-transmil-VIM/MamMIL2.yaml --gpus $GPUS --fold 0 --seed 1
/usr/local/bin/python3 train.py --stage test --config configs/Camelyon16-transmil-VIM/MamMIL2.yaml --gpus $GPUS --fold 0 --seed 1

/usr/local/bin/python3 train.py --stage train --config configs/Camelyon16-transmil-VMAMBA/MamMIL2.yaml --gpus $GPUS --fold 0 --seed 1
/usr/local/bin/python3 train.py --stage test --config configs/Camelyon16-transmil-VMAMBA/MamMIL2.yaml --gpus $GPUS --fold 0 --seed 1

for fold in {0..4}
do
    /usr/local/bin/python3 train.py --stage train --config configs/TCGA-LUAD_survival/MamMIL2.yaml --gpus $GPUS --fold $fold --seed 1
    /usr/local/bin/python3 train.py --stage test --config configs/TCGA-LUAD_survival/MamMIL2.yaml --gpus $GPUS --fold $fold --seed 1
done

for fold in {0..4}
do
    /usr/local/bin/python3 train.py --stage train --config configs/TCGA-LUAD_survival-VIM/MamMIL2.yaml --gpus $GPUS --fold $fold --seed 1
    /usr/local/bin/python3 train.py --stage test --config configs/TCGA-LUAD_survival-VIM/MamMIL2.yaml --gpus $GPUS --fold $fold --seed 1
done

for fold in {0..4}
do
    /usr/local/bin/python3 train.py --stage train --config configs/TCGA-LUAD_survival-VMAMBA/MamMIL2.yaml --gpus $GPUS --fold $fold --seed 1
    /usr/local/bin/python3 train.py --stage test --config configs/TCGA-LUAD_survival-VMAMBA/MamMIL2.yaml --gpus $GPUS --fold $fold --seed 1
done
