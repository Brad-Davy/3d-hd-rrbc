directory='results/Ra_8-00e+08_Ek_1-00e-06_Pr_7-0_N_192_Asp_1-0/'


for VARIABLE in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
    python3 real-forces.py --dir=$directory --mask=$VARIABLE --snap_t=1
    echo ' '
done

