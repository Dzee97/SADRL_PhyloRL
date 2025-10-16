#!/bin/bash

# pars
./raxml-ng --start --msa ../dataset/050_258_p__Basidiomycota_c__Agaricomycetes_o__Boletales.fasta --model GTR+I+G --tree pars{10} --redo --prefix pars/start
./raxml-ng --evaluate --msa ../dataset/050_258_p__Basidiomycota_c__Agaricomycetes_o__Boletales.fasta --model GTR+I+G --tree pars/start.raxml.startTree --redo --prefix pars/eval

# rand
./raxml-ng --start --msa ../dataset/050_258_p__Basidiomycota_c__Agaricomycetes_o__Boletales.fasta --model GTR+I+G --tree rand{10} --redo --prefix rand/start
./raxml-ng --evaluate --msa ../dataset/050_258_p__Basidiomycota_c__Agaricomycetes_o__Boletales.fasta --model GTR+I+G --tree rand/start.raxml.startTree --redo --prefix rand/eval
