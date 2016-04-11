#!/bin/bash

file("")


output="output.txt"
> $output

N=128
P=2
for i in 2 4 8; do
	mpirun -np $P ./poisson $N > $Variabel
	fprintf >> $Variable 
done

