#!/bin/bash

case $1 in
    '-del')
        rm -rf scaler.pkl sudret.pdf ./models/* .vscode/
        ;;
    '-w')
        rm -rf scaler.pkl hyper_params.json sudret.pdf ./models/* ./data/* .vscode/
        ;;
    '-ls')
        tree -L 3 --dirsfirst -I 'models|data|scaler.pkl|sudret.pdf|manage|__pycache__/'
        ;;
    *)
        echo 'Opções: Listar (-ls) Limpar (-del) Resetar (-w)'
        ;;
esac