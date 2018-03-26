#!/bin/bash
gnuplot -e "set ylabel \"MegaFlops\";set xlabel \"Non-zeros\";set datafile separator ' '; set term png; \
set output \"./outputs/csr.png\"; plot \"./outputs/sequential_sorted.csv\" using 1:2 title 'Sequential' with linespoints ls 3 lw 2 lc rgb '#0060ad' pt 9 ps 1, \
     \"./outputs/openmp_sorted.csv\" using 1:2 title 'OpenMP' with linespoints ls 2 lw 2 lc rgb '#dd181f' pt 5 ps 1, \
     \"./outputs/cuda_sorted.csv\" using 1:2 title 'CUDA' with linespoints ls 1 lw 2 lc rgb '#FFA500' pt 7 ps 1"

gnuplot -e "set ylabel \"MegaFlops\";set xlabel \"Non-zeros\";set datafile separator ' '; set term png; \
 set output \"./outputs/ellpack.png\"; plot \"./outputs/sequential_sorted.csv\" using 1:3 title 'Sequential' with linespoints ls 3 lw 2 lc rgb '#0060ad' pt 9 ps 1, \
     \"./outputs/openmp_sorted.csv\" using 1:3 title 'OpenMP' with linespoints ls 2 lw 2 lc rgb '#dd181f' pt 5 ps 1, \
     \"./outputs/cuda_sorted.csv\" using 1:3 title 'CUDA' with linespoints ls 1 lw 2 lc rgb '#FFA500' pt 7 ps 1"

gnuplot -e "set ylabel \"MegaFlops\";set xlabel \"Non-zeros\";set datafile separator ' '; set term png; \
set output \"./outputs/csr_wo_cuda.png\"; plot \"./outputs/sequential_sorted.csv\" using 1:2 title 'Sequential' with linespoints ls 3 lw 2 lc rgb '#0060ad' pt 9 ps 1, \
     \"./outputs/openmp_sorted.csv\" using 1:2 title 'OpenMP' with linespoints ls 2 lw 2 lc rgb '#dd181f' pt 5 ps 1"

gnuplot -e "set ylabel \"MegaFlops\";set xlabel \"Non-zeros\";set datafile separator ' '; set term png; \
 set output \"./outputs/ellpack_wo_cuda.png\"; plot \"./outputs/sequential_sorted.csv\" using 1:3 title 'Sequential' with linespoints ls 3 lw 2 lc rgb '#0060ad' pt 9 ps 1, \
     \"./outputs/openmp_sorted.csv\" using 1:3 title 'OpenMP' with linespoints ls 2 lw 2 lc rgb '#dd181f' pt 5 ps 1"