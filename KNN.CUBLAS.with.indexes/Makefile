all:
	nvcc -arch=sm_20 knn_cublas_with_indexes.cu -lcuda -lcublas -o knn_cublas_with_indexes

run:
	./knn_cublas_with_indexes uniform11_d100_n100000_float32.bin uniform11_d100_n100_float32.bin
	diff idx.txt sklearn-knn-idx_k100_nq100_nr100000.txt
