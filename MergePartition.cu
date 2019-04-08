#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <pthread.h>
#define MAX 1024*1024*1024
#define N_STREAMS 4
#define N_THREADS 4

typedef struct {
	int **inp1;
	int **inp2;
	int **out;
	int psize;
	int noparts;
} _compdata;

struct args {
	_compdata *data;
	int t_no;
};
double getTime(void) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_usec/1e6 + tv.tv_sec;
}

void checkStatus(cudaError_t st, const char *method) {
	if(st != cudaSuccess) {
		printf("Error: Error while doing %s\n", method);
		exit(-1);
	}
}

void cpuMerge(int *a ,int *b, int *res, int size) {
	int i = 0, j = 0, z = 0;
	while (z < size * 2) {
		if((i < size) && (j < size)) {
			if(a[i] <= b[j]) {
				res[z] = a[i];
				i++;
			}
			else {
				res[z] = b[j];
				j++;
			}
		}
		else if(i < size) {
			res[z] = a[i];
			i++;
		}
		else if (j < size) {
			res[z] = b[j];
			j++;
		}
		z++;
	}
}

void* tCpuMerge(void *args) {
	int i;
	struct args *part = (struct args *)args;
	int chunksize = part->data->noparts/N_THREADS;
	for(i = part->t_no * chunksize; i < (part->t_no + 1) * chunksize; i++) {
		cpuMerge(part->data->inp1[i], part->data->inp2[i], part->data->out[i], part->data->psize);
	}
	return NULL;
}
__device__ int binSearch(int elem, int *Sa, int n, int *is_mid) {

	int low = 0, high = n - 1, mid;
	*is_mid = 0;
	if(elem > Sa[high]) {
		return high + 1;
	}
	else if(elem < Sa[low]) {
		return low;
	}
	while(low <= high) {
		mid = (low + high)/2;
		if(elem < Sa[mid]) {
			high = mid - 1;
		}
		else if(elem > Sa[mid]) {
			low = mid + 1;
		}
		else {
			*is_mid = 1;
			//printf("match %d\n", elem);
			break;
		}
	}
	
	return mid;
}
__global__ void d_merge(int *A, int *B, int *C, int n) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int is_mid;
	int ret;
	//find final location for A[pos] and B[pos]
	if(pos < n) {
		ret = binSearch(A[pos], B, n, &is_mid);
		C[pos + ret + is_mid] = A[pos];
		ret = binSearch(B[pos], A, n, &is_mid);
		C[pos + ret] = B[pos];
	}

}

int comp(const void *a, const void *b) {
	int l = *(const int*)a;
	int r = *(const int*)b;
	if(l > r) return 1;
	if(l < r) return -1;
	return 0;
}

int check_sorted(_compdata *data) {
	int i, j;
	for(i = 0; i < data->noparts; i++) {
		for(j = 0; j < (2 * data->psize) - 1; j++) {
			//printf("%d %d\n", j, data->out[i][j]);
			if(data->out[i][j] > data->out[i][j+1]) {
				printf("Value incorrect at pos %d %d\n", j + 1, data->out[i][j+1]);
				return 0;
			}
		}
	}
	return 1;
}
void initCompData(_compdata *data) {
	int i, j;
	int bytes = data->psize * sizeof(int);
	cudaError_t status;
	data->inp1 = (int **) malloc(sizeof(int *) * data->noparts);
	data->inp2 = (int **) malloc(sizeof(int *) * data->noparts);
	data->out = (int **) malloc(sizeof(int *) * data->noparts);
	for(i = 0; i < data->noparts; i++) {
		status = cudaHostAlloc((void **)&data->inp1[i], bytes, cudaHostAllocDefault);
		checkStatus(status, "cudaHostAlloc");
		status = cudaHostAlloc((void **)&data->inp2[i], bytes, cudaHostAllocDefault);
		checkStatus(status, "cudaHostAlloc");
		status = cudaHostAlloc((void **)&data->out[i], 2 * bytes, cudaHostAllocDefault);
		checkStatus(status, "cudaHostAlloc");
		for(j = 0; j < data->psize; j++) {
			data->inp1[i][j] = data->psize*i + j;
			data->inp2[i][j] = data->psize*i + j + 2;
		}
	}
	return;
}

void deviceMalloc(int **d_inp1, int **d_inp2, int **d_out, int psize) {
	int i;
	cudaError_t status;
	for(i = 0; i < N_STREAMS; i++) {
		status = cudaMalloc((void **)&d_inp1[i], psize * sizeof(int));
		checkStatus(status, "cudaMalloc");
		status = cudaMalloc((void **)&d_inp2[i], psize * sizeof(int));
		checkStatus(status, "cudaMalloc");
		status = cudaMalloc((void **)&d_out[i], 2 * psize * sizeof(int));
		checkStatus(status, "cudaMalloc");
	}

}
int main(int argc, char *argv[])
{
	cudaError_t status;
	_compdata data;	
	struct args part[N_THREADS];
	pthread_t tids[N_THREADS];
	int i, j;
	int bytes;
	int *d_inp1[N_STREAMS], *d_inp2[N_STREAMS], *d_out[N_STREAMS];
	cudaStream_t streams[N_STREAMS];
	double ts_a, ts_b, ts_c;

	if(argc != 3) {
		printf("Required: partition size, number of partitions\n");
		return -1;
	}
	data.psize = atoi(argv[1]);
	data.noparts = atoi(argv[2]);
	bytes = data.psize * sizeof(int);
	//initilization
	initCompData(&data);
	deviceMalloc(d_inp1, d_inp2, d_out, data.psize);
	dim3 block(32);
	dim3 grid((data.psize + block.x - 1)/block.x);

	//create streams
	for(i = 0; i < N_STREAMS; i++) {
		cudaStreamCreate(&streams[i]);
	}

	ts_a = getTime();
	//launch transfer and kernels in streams
	assert(data.noparts % N_STREAMS == 0);
	for(i = 0; i < data.noparts; i += N_STREAMS) {
		for(j = 0; j < N_STREAMS; j++) {
			cudaMemcpyAsync(d_inp1[j], data.inp1[i+j], bytes, cudaMemcpyHostToDevice, streams[j]);
			cudaMemcpyAsync(d_inp2[j], data.inp2[i+j], bytes, cudaMemcpyHostToDevice, streams[j]);
			//launch kernel
			d_merge <<<grid, block, 0, streams[j]>>> (d_inp1[j], d_inp2[j], d_out[j], data.psize);
			cudaMemcpyAsync(data.out[i+j], d_out[j], 2 * bytes, cudaMemcpyDeviceToHost, streams[j]);
		}

	}
	
	status = cudaDeviceSynchronize();
	checkStatus(status, "cudaDeviceSync");
	ts_b = getTime();
	if(check_sorted(&data) == 0) {
		printf("Result GPU incorrect\n");
	}
	for(i = 0; i < N_THREADS; i++) {
		part[i].data = &data;
		part[i].t_no = i;
		pthread_create(&tids[i], NULL, &tCpuMerge, (void *)&part[i]);
	}
	for(i = 0; i < N_THREADS; i++) {
		pthread_join(tids[i], NULL);
	}
	ts_c = getTime();
	if(check_sorted(&data) == 0) {
		printf("Result CPU incorrect\n");
	}
	printf("Execution time Streaming %d GPU %.6f %d threads CPU %.6f\n",N_STREAMS, ts_b - ts_a, N_THREADS, ts_c - ts_b);

	cudaDeviceReset();

	return 0;
}
