all: sequential openmp cuda

sequential:
	g++ -O3 -std=c++11 -Wall -fopenmp src/sequential.cpp src/*/*.cpp -o bin/sequential
openmp:
	g++ -O3 -std=c++11 -Wall -fopenmp -lpthread src/openmp.cpp src/*/*.cpp -o bin/openmp
cuda:
	nvcc -O3 -std=c++11 -Xcompiler -fopenmp -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__ src/cuda.cpp src/*/*.cpp src/cuda.cu -o bin/cuda
run:
	rm -f outputs/*.csv
	bin/sequential matrices/adder_dcop_32.mtx
	bin/openmp matrices/adder_dcop_32.mtx
	bin/cuda matrices/adder_dcop_32.mtx
runall:
	rm -f outputs/*.csv
	for m in $(wildcard matrices/*.mtx); \
		do printf "Running: ".$$m && printf "  Sequential " && bin/sequential $$m && printf " >>  OpenMP " && bin/openmp $$m && printf " >>  Cuda " && bin/cuda $$m && printf "  Done!\n"; \
	done \
plotcsr:
	gnuplot -e "set ylabel \"MegaFlops\";set xlabel \"Matrices\";set datafile separator ' '; set format x \"%.2g\"; set xtics 1; set term png; set output \"./outputs/csr.png\"; plot \"./outputs/sequential.csv\" using 1:2 title 'Sequential' with linespoints ls 3 lw 2 lc rgb '#0060ad' pt 9 ps 1.5, \
     \"./outputs/openmp.csv\" using 1:2 title 'OpenMP' with linespoints ls 2 lw 2 lc rgb '#dd181f' pt 5 ps 1.5, \
     \"./outputs/cuda.csv\" using 1:2 title 'CUDA' with linespoints ls 1 lw 2 lc rgb '#FFA500' pt 7 ps 1.5"
plotellpack:
	gnuplot -e "set ylabel \"MegaFlops\";set xlabel \"Matrices\";set datafile separator ' '; set format x \"%.2g\"; set xtics 1; set term png; set output \"./outputs/ellpack.png\"; plot \"./outputs/sequential.csv\" using 1:3 title 'Sequential' with linespoints ls 3 lw 2 lc rgb '#0060ad' pt 9 ps 1.5, \
     \"./outputs/openmp.csv\" using 1:3 title 'OpenMP' with linespoints ls 2 lw 2 lc rgb '#dd181f' pt 5 ps 1.5, \
     \"./outputs/cuda.csv\" using 1:3 title 'CUDA' with linespoints ls 1 lw 2 lc rgb '#FFA500' pt 7 ps 1.5"

