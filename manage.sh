#!/bin/bash

case $1 in
    '-w')
        rm -rf scaler.pkl sudret.pdf ./models/* .vscode/
        ;;
    '--lista')
        tree -L 3 --dirsfirst -I 'models|data|scaler.pkl|sudret.pdf|manage|__pycache__/'
        ;;
    *)
        echo 'Opções: Listar (--lista) Resetar (-w)'
        ;;
esac