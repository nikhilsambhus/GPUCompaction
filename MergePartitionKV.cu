#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#define MAX 1024*1024*1024
#define N_STREAMS 4
#define N_THREADS 8
#define KSIZE 16
//#define VSIZE (512-16)
#define VSIZE (32-16)
//#define VSIZE (128-16)

typedef struct {
	char key[KSIZE];
	char value[VSIZE];
} _kv;

typedef struct {
	_kv **inp1;
	_kv **inp2;
	_kv **out;
	int psize;
	int noparts;
} _compdata;

struct args {
	_compdata *data;
	int t_no;
};

struct partStart {
	int ai, bi;
};
//str functions for cuda
__device__ void c_strcpy(char *dest, char *src) {
	int i = 0;
	while(src[i] != '\0') {
		dest[i] = src[i];
		i++;
	}
	dest[i] = '\0';
}

__device__ void c_memcpy(void *dest, void *src, int size) {
	//memcpy(dest, src, size);
	//return;
	int i;
	uint64_t *dp = (uint64_t *)dest, *sp = (uint64_t *)src;
	for(i = 0; i < size; i+=8) {
		*dp = *sp;
	}
}	
__device__ int c_atoi(char *str) {
	int i = 0, ret = 0;
	while(str[i] != '\0') {
		ret = ret * 10 + str[i] - '0';
		i++;
	}
	return ret;
}

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

void cpuMerge(_kv *a, _kv *b, _kv *res, int size) {
	int i = 0, j = 0, z = 0;
	while (z < size * 2) {
		if((i < size) && (j < size)) {
			if(atoi(a[i].key) <= atoi(b[j].key)) {
				memcpy(&res[z], &a[i], KSIZE+VSIZE);
				i++;
			}
			else {
				memcpy(&res[z], &b[j], KSIZE+VSIZE);
				j++;
			}
		}
		else if(i < size) {
			memcpy(&res[z], &a[i], KSIZE+VSIZE);
			i++;
		}
		else if (j < size) {
			memcpy(&res[z], &b[j], KSIZE+VSIZE);
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
__device__ int binSearch(char *elem, _kv *Sa, int n, int *is_mid) {

	int low = 0, high = n - 1, mid;
	*is_mid = 0;
	int elem_i, mid_i;
	elem_i = c_atoi(elem);
	if(elem_i > c_atoi(Sa[high].key)) {
		return high + 1;
	}
	else if(elem_i < c_atoi(Sa[low].key)) {
		return low;
	}
	while(low <= high) {
		mid = (low + high)/2;
		elem_i = c_atoi(elem);  
		mid_i = c_atoi(Sa[mid].key);
		if(elem_i < mid_i) {
			high = mid - 1;
		}
		else if(elem_i > mid_i) {
			low = mid + 1;
		}
		else {
			*is_mid = 1;
			break;
		}
	}
	
	return mid;
}

__global__ void d_binS_merge(_kv *A, _kv *B, _kv *C, int n) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int is_mid = 0;
	int ret;
	//find final location for A[pos] and B[pos]
	if(pos < n) {
		ret = 0;
		ret = binSearch(A[pos].key, B, n, &is_mid);
		c_strcpy(C[pos + ret + is_mid].key, A[pos].key);
		memcpy(C[pos + ret + is_mid].value, A[pos].value, VSIZE);

		ret = binSearch(B[pos].key, A, n, &is_mid);
		c_strcpy(C[pos + ret].key, B[pos].key);
		memcpy(C[pos + ret].value, A[pos].value, VSIZE);
	}

}


__device__ struct partStart diagInter(_kv *A, _kv *B, int n, int pindex, int parts) {
	int diag = ((n/parts) << 1) * pindex;
	int begin = ((diag - n ) > 0 ? diag - n: 0);
	int end = (diag > n ? n : diag);
	int aKey, bKey;
	struct partStart ABstart;
	while(begin < end) {
		int mid = (begin + end) >> 1;
		aKey = c_atoi(A[mid].key);
		bKey = c_atoi(B[diag - 1 - mid].key);
		if(aKey < bKey) {
			begin = mid + 1;
		}
		else {
			end = mid;
		}
	}

	ABstart.ai = begin;
	ABstart.bi = diag - begin;
	//printf("pindex %d ai %d bi %d\n", pindex, ABstart.ai, ABstart.bi);
	return ABstart;

}

__device__ void merge(_kv *A, int ai, _kv *B, int bi, _kv *C, int ci, int n, int len) {
	int i = ai, j = bi, z = ci;
	while (z < ci + len) {
		if((i < n) && (j < n)) {
			if(c_atoi(A[i].key) <= c_atoi(B[j].key)) {
				c_memcpy(&C[z], &A[i], KSIZE + VSIZE);
				i++;
			}
			else {
				c_memcpy(&C[z], &B[j], KSIZE + VSIZE);
				j++;
			}
		}
		else if(i < n) {
			c_memcpy(&C[z], &A[i], KSIZE + VSIZE);
			i++;
		}
		else if (j < n) {
			c_memcpy(&C[z], &B[j], KSIZE + VSIZE);
			j++;
		}
		z++;
	}
}
__global__ void d_pathMerge(_kv *A, _kv *B, _kv *C, int n, int parts) {
	//return;
	int lenPart = ((n << 1)/parts);
	//lenPart = 0;
	struct partStart ABstart;
	int Cstart;
	int pindex = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(pindex < parts) {
		ABstart = diagInter(A, B, n, pindex, parts);
		Cstart = (n << 1)/parts * pindex;
		merge(A, ABstart.ai, B, ABstart.bi, C, Cstart, n, lenPart);
	}
}
void pathMerge(_kv *A, _kv *B, _kv *C, int n, int parts, cudaStream_t stream) {
	dim3 block(32);
	dim3 grid((parts + block.x - 1)/block.x);
	d_pathMerge<<<grid, block, 0, stream>>> (A, B, C, n, parts);
}
int check_sorted(_compdata *data) {
	int i, j;
	for(i = 0; i < data->noparts; i++) {
		for(j = 0; j < (2 * data->psize) - 1; j++) {
			//printf("%d %s\n", j, data->out[i][j].key);
			if(atoi(data->out[i][j].key) > atoi(data->out[i][j+1].key)) {
				printf("Value incorrect at pos %d %s\n", j + 1, data->out[i][j+1].key);
				return 0;
			}
		}
		//printf("%d %s\n", j, data->out[i][j].key);
	}
	return 1;
}

int cmpF(const void *a, const void *b) {
	int v1 = *(int *)a;
	int v2 = *(int *)b;
	if(v1 < v2) return -1;
	else if(v1 > v2) return 1;
	else return 0;
}	

void initCompData(_compdata *data) {
	int i, j;
	int bytes = data->psize * sizeof(_kv);
	int *randSet1, *randSet2;
	randSet1 = (int *)malloc(data->psize * sizeof(int));
	randSet2 = (int *)malloc(data->psize * sizeof(int));
	srand(100);
	for (i = 0; i < data->psize; i++) {
		randSet1[i] = rand();
		randSet2[i] = rand();
	}
	qsort(randSet1, data->psize, sizeof(int), cmpF);
	qsort(randSet2, data->psize, sizeof(int), cmpF);
	cudaError_t status;
	data->inp1 = (_kv **) malloc(sizeof(_kv *) * data->noparts);
	data->inp2 = (_kv **) malloc(sizeof(_kv *) * data->noparts);
	data->out = (_kv **) malloc(sizeof(_kv *) * data->noparts);
	for(i = 0; i < data->noparts; i++) {
		status = cudaHostAlloc((void **)&data->inp1[i], bytes, cudaHostAllocDefault);
		checkStatus(status, "cudaHostAlloc");
		status = cudaHostAlloc((void **)&data->inp2[i], bytes, cudaHostAllocDefault);
		checkStatus(status, "cudaHostAlloc");
		status = cudaHostAlloc((void **)&data->out[i], 2 * bytes, cudaHostAllocDefault);
		checkStatus(status, "cudaHostAlloc");
		for(j = 0; j < data->psize; j++) {
			sprintf(data->inp1[i][j].key, "%015d", randSet1[j]);
			memset(data->inp1[i][j].value, 'x', VSIZE);
			sprintf(data->inp2[i][j].key, "%015d", randSet2[j]);
			memset(data->inp2[i][j].value, 'x', VSIZE);
			//printf("%s %s\n", data->inp1[i][j].key, data->inp2[i][j].key);
		}
	}

	free(randSet1);
	free(randSet2);
	printf("Initilization done\n");
	return;
}

void deviceMalloc(_kv **d_inp1, _kv **d_inp2, _kv **d_out, int psize) {
	int i;
	cudaError_t status;
	for(i = 0; i < N_STREAMS; i++) {
		status = cudaMalloc((void **)&d_inp1[i], psize * sizeof(_kv));
		checkStatus(status, "cudaMalloc");
		status = cudaMalloc((void **)&d_inp2[i], psize * sizeof(_kv));
		checkStatus(status, "cudaMalloc");
		status = cudaMalloc((void **)&d_out[i], 2 * psize * sizeof(_kv));
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
	_kv *d_inp1[N_STREAMS], *d_inp2[N_STREAMS], *d_out[N_STREAMS];
	cudaStream_t streams[N_STREAMS];
	double ts_a, ts_b, ts_c, ts_d;

	if(argc != 3) {
		printf("Required: partition size, number of partitions\n");
		return -1;
	}
	data.psize = atoi(argv[1]);
	data.noparts = atoi(argv[2]);
	bytes = data.psize * sizeof(_kv);
	printf("Total size (2 inputs + 1 output) %.2f MB\n", (bytes * 4.0 * data.noparts)/1024/1024);
	printf("Size of one partition (2 inputs + 1 output) %.2f MB\n", (bytes * 4.0)/1024/1024);
	//initilization
	initCompData(&data);
	deviceMalloc(d_inp1, d_inp2, d_out, data.psize);
	dim3 block(1024);
	dim3 grid((data.psize + block.x - 1)/block.x);

	//create streams
	for(i = 0; i < N_STREAMS; i++) {
		cudaStreamCreate(&streams[i]);
	}

	assert(data.noparts % N_STREAMS == 0);
	ts_a = getTime();
	//launch transfer and kernels in streams
	for(i = 0; i < data.noparts; i += N_STREAMS) {
		for(j = 0; j < N_STREAMS; j++) {
			cudaMemcpyAsync(d_inp1[j], data.inp1[i+j], bytes, cudaMemcpyHostToDevice, streams[j]);
			cudaMemcpyAsync(d_inp2[j], data.inp2[i+j], bytes, cudaMemcpyHostToDevice, streams[j]);
			//launch kernel
			//d_binS_merge <<<grid, block, 0, streams[j]>>> (d_inp1[j], d_inp2[j], d_out[j], data.psize);
			pathMerge(d_inp1[j], d_inp2[j], d_out[j], data.psize, (data.psize << 1), streams[j]); 
			cudaMemcpyAsync(data.out[i+j], d_out[j], 2 * bytes, cudaMemcpyDeviceToHost, streams[j]);
		}

	}
	status = cudaDeviceSynchronize();
	ts_b = getTime();
	checkStatus(status, "cudaDeviceSync");
	if(check_sorted(&data) == 0) {
		printf("Result GPU incorrect\n");
	}

	ts_c = getTime();
	for(i = 0; i < N_THREADS; i++) {
		part[i].data = &data;
		part[i].t_no = i;
		pthread_create(&tids[i], NULL, &tCpuMerge, (void *)&part[i]);
	}
	for(i = 0; i < N_THREADS; i++) {
		pthread_join(tids[i], NULL);
	}
	ts_d = getTime();
	if(check_sorted(&data) == 0) {
		printf("Result CPU incorrect\n");
	}
	printf("Execution time Streaming %d GPU %.6f %d threads CPU %.6f\n",N_STREAMS, ts_b - ts_a, N_THREADS, ts_d - ts_c);

	cudaDeviceReset();

	return 0;
}
