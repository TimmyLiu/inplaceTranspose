#define _CRT_SECURE_NO_WARNINGS

#include <assert.h>
#include <iostream>
#include <CL/cl.h>


const char PLATFORM_NAME[] = "AMD Accelerated Parallel Processing";
const char DEVICE_NAME[] = "Fiji";
//const char BUILD_OPTIONS[] = "-cl-std=CL2.0";
const char BUILD_OPTIONS[] = "";
//const char KERNEL_SOURCE[] = "C:\\Users\\timmy\\Documents\\inplaceTranspose_client\\inplaceTransposeNonSquare_one2Two_client_src\\clfft.kernel.Transpose2_real_1024_2048.cl";
//const char KERNEL_SOURCE[] = "C:\\Users\\timmy\\Documents\\inplaceTranspose_client\\inplaceTransposeNonSquare_one2Two_client_src\\TransposeNonSquare4by8.cl";
const char KERNEL_SOURCE[] = "C:\\Users\\timmy\\Documents\\inplaceTranspose_client\\inplaceTransposeNonSquare_one2Two_client_src\\TransposeNonSquare1024by2048.cl";


cl_kernel
createKernel(
	const char* source,
	cl_context context,
	const char* options,
	cl_int* error)
{

	cl_int err;
	cl_device_id device;
	cl_program program;
	cl_kernel kernel;
	size_t logSize;
	char *log;

	err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(device), &device, NULL);
	if (err != CL_SUCCESS) {
		if (error != NULL) {
			*error = err;
		}
		return NULL;
	}

	program = clCreateProgramWithSource(context, 1, &source, NULL, error);
	if (program == NULL) {
		return NULL;
	}

	err = clBuildProgram(program, 1, &device, options, NULL, NULL);
	if (err != CL_SUCCESS) {
		logSize = 0;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
		log = (char*)calloc(1, logSize + 1);
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
		printf("=== Build log ===\n%s\n", log);
		free(log);
		clReleaseProgram(program);
		if (error != NULL) {
			*error = err;
		}
		return NULL;
	}

	kernel = NULL;
	err = clCreateKernelsInProgram(program, 1, &kernel, NULL);
	clReleaseProgram(program);
	if (error != NULL) {
		*error = err;
	}
	return kernel;
}

char*
loadFile(const char* path)
{
	FILE *f;
	long size;
	char *text;

	f = fopen(path, "rb");
	if (f == NULL) {
		return NULL;
	}

	if (fseek(f, 0, SEEK_END) != 0) {
		fclose(f);
		return NULL;
	}
	size = ftell(f);
	if (size == -1) {
		fclose(f);
		return NULL;
	}
	if (fseek(f, 0, SEEK_SET) != 0) {
		fclose(f);
		return NULL;
	}

	text = (char*)calloc(size + 1, 1);
	if (text == NULL) {
		fclose(f);
		return NULL;
	}

	if (fread(text, 1, size, f) == 0) {
		free(text);
		fclose(f);
		return NULL;
	}
	fclose(f);
	return text;
}

cl_platform_id
getPlatform(const char *name)
{
	cl_int err;
	cl_uint nrPlatforms, i;
	cl_platform_id *list, platform;
	char platformName[64];

	err = clGetPlatformIDs(0, NULL, &nrPlatforms);
	if (err != CL_SUCCESS) {
		return NULL;
	}

	list = (cl_platform_id*)calloc(nrPlatforms, sizeof(*list));
	if (list == NULL) {
		return NULL;
	}

	err = clGetPlatformIDs(nrPlatforms, list, NULL);
	if (err != CL_SUCCESS) {
		free(list);
		return NULL;
	}

	platform = NULL;
	for (i = 0; i < nrPlatforms; i++) {
		err = clGetPlatformInfo(list[i], CL_PLATFORM_NAME,
			sizeof(platformName), platformName, NULL);
		if ((err == CL_SUCCESS) && (strcmp(platformName, name) == 0)) {
			platform = list[i];
			break;
		}
	}

	free(list);
	return platform;
}

cl_device_id
getDevice(
	cl_platform_id platform,
	const char *name)
{

	cl_int err;
	cl_uint nrDevices, i;
	cl_device_id *list, device;
	char deviceName[64];

	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &nrDevices);
	if (err != CL_SUCCESS) {
		return NULL;
	}
	list = (cl_device_id*)calloc(nrDevices, sizeof(*list));
	if (list == NULL) {
		return NULL;
	}

	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, nrDevices, list, NULL);
	if (err != CL_SUCCESS) {
		free(list);
		return NULL;
	}

	device = NULL;
	for (i = 0; i < nrDevices; i++) {
		err = clGetDeviceInfo(list[i], CL_DEVICE_NAME,
			sizeof(deviceName), deviceName, NULL);
		if ((err == CL_SUCCESS) && (strcmp(deviceName, name) == 0)) {
			device = list[i];
			break;
		}
	}

	free(list);
	return device;
}

int main(int argc, char **argv)
{
	//single precision real number
	//row major m rows by n columns
	int performance_level = atoi(argv[1]);
	int m = atoi(argv[2]);//m and n should be mod 32; 2048 by 1024
	int n = atoi(argv[3]);
	int batchSize = atoi(argv[4]);

	//malloc input data
	float *CPU_A = (float*)malloc(m*n*batchSize*sizeof(float));
	for (int k = 0; k < batchSize; k++)
	{
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				CPU_A[k*m*n + i*n + j] = i*n + j;
			}
		}
	}
	/*
	for (int k = 0; k < batchSize; k++)
	{
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				std::cout<< CPU_A[k*m*n + i*n + j] << " ";
			}
			std::cout << std::endl;
		}
	}
	*/
	//for (int i = 0; i < m*n; i++)
	//	std::cout << CPU_A[i] << " ";

	//init OpenCL 
	cl_int err;
	cl_platform_id platform;
	cl_device_id device;
	cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
	cl_context context;
	cl_command_queue queue;
	cl_kernel kernel;
	cl_event event;
	char *source;

	platform = getPlatform(PLATFORM_NAME);
	assert(platform != NULL);
	device = getDevice(platform, DEVICE_NAME);
	assert(device != NULL);
	props[1] = (cl_context_properties)platform;
	context = clCreateContext(props, 1, &device, NULL, NULL, &err);
	assert(context != NULL);
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
	assert(queue != NULL);

	cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_WRITE,
		(m * n * batchSize) * sizeof(*CPU_A), NULL, &err);
	assert(bufA != NULL);

	//move memory from host to device
	err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
		(m*n*batchSize) * sizeof(*CPU_A), CPU_A,
		0, NULL, NULL);

	//compile kernel
	source = loadFile(KERNEL_SOURCE);
	assert(source != NULL);
	kernel = createKernel(source, context, BUILD_OPTIONS, &err);
	assert(kernel != NULL);


	//launch kernel
	size_t localWorkSize[1] = { 256 };

	size_t globalWorkSize[1] = { 99878 };//we get this number from permutation calculation; most threads work on 21 elements 99878+2

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
	assert(err == CL_SUCCESS);
	/*
	err = clSetKernelArg(kernel, 1, sizeof(cl_uint), &m);
	assert(err == CL_SUCCESS);
	err = clSetKernelArg(kernel, 2, sizeof(cl_uint), &num_wg);
	assert(err == CL_SUCCESS);
	*/

	if (performance_level == 0)
	{
		//check result
		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
			globalWorkSize, localWorkSize, 0, NULL, &event);
		assert(err == CL_SUCCESS);
		clWaitForEvents(1, &event);
		assert(err == CL_SUCCESS);
	}
	else if (performance_level == 1)
	{
		//check kernel performance
		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
			globalWorkSize, localWorkSize, 0, NULL, &event);
		assert(err == CL_SUCCESS);
		clWaitForEvents(1, &event);
		assert(err == CL_SUCCESS);
		cl_ulong start, end;
		cl_ulong KernelTime = 0;
		int iteration = 20;
		for (int i = 0; i < iteration; i++)
		{
			event = NULL;
			err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
				globalWorkSize, localWorkSize, 0, NULL, &event);

			assert(err == CL_SUCCESS);
			clWaitForEvents(1, &event);
			assert(err == CL_SUCCESS);


			err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
				sizeof(start), &start, NULL);
			err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
				sizeof(end), &end, NULL);
			KernelTime += (end - start);

		}

		//KernelTime is in ns
		size_t peakGBs = 512;
		size_t KernelGBs = 2 * sizeof(float) * m * n * batchSize / (KernelTime / iteration);
		std::cout << " GBs: " << KernelGBs << " GBs" << std::endl;
		float efficiency = ((float)KernelGBs) / (float)peakGBs;
		std::cout << " efficiency: " << efficiency * 100 << "%" << std::endl;
	}

	//move memory from device to host
	err = clEnqueueReadBuffer(queue, bufA, CL_TRUE, 0,
		(batchSize*m*n) * sizeof(*CPU_A), CPU_A,
		0, NULL, NULL);
	assert(err == CL_SUCCESS);

	//print result
	/*
	for (int k = 0; k < batchSize; k++)
	{
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < m; j++)
			{
				std::cout << CPU_A[k*m*n + i*m + j] << " ";
			}
			std::cout << std::endl;
		}
	}
	*/
	//std::cout << std::endl;
	//for (int i = 0; i < m*n; i++)
	//	std::cout << CPU_A[i] << " ";

	if (performance_level == 0)
	{
		//check result
		int error = 0;
		for (int k = 0; k < batchSize; k++)
		{
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < m; j++)
				{
					float temp = CPU_A[i*m + j];
					if (CPU_A[k*m*n + i*m + j] != (j*n + i))
					{
						std::cout << "batch idex = " << k << " ";
						std::cout << "CPU_A[" << i << "*n + " << j << "] = " << CPU_A[i*n + j] << std::endl;
						error = 1;
						break;
					}
				}
			}
		}

		if (error == 0)
		{
			std::cout << "correstness passed." << std::endl;
		}
		else
		{
			std::cout << "correctness failed." << std::endl;
		}
	}

	//releasing the objects
	err = clReleaseMemObject(bufA);
	err = clReleaseEvent(event);
	err = clReleaseKernel(kernel);
	err = clReleaseCommandQueue(queue);
	err = clReleaseContext(context);
	free(CPU_A);
}
