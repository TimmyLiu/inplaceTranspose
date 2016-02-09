#define _CRT_SECURE_NO_WARNINGS

#include <assert.h>
#include <iostream>
#include <complex>
#include <CL/cl.h>


const char PLATFORM_NAME[] = "AMD Accelerated Parallel Processing";
const char DEVICE_NAME[] = "Fiji";
const char BUILD_OPTIONS[] = "";
//const char BUILD_OPTIONS[] = "";
const char KERNEL_SOURCE1[] = "C:\\Users\\timmy\\Documents\\inplaceTranspose_client\\inplaceTranseposeNoneSquare_one2Two_2pass_src\\TransposeNonSquare1stPass.cl";
const char KERNEL_SOURCE2[] = "C:\\Users\\timmy\\Documents\\inplaceTranspose_client\\inplaceTranseposeNoneSquare_one2Two_2pass_src\\TransposeNonSquare2ndPass.cl";
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
	int m = atoi(argv[2]);//m and n should be mod 32
	int n = atoi(argv[3]);
	int batchSize = atoi(argv[4]);
	//n should be twice as m for now.
	if (n != 2 * m)
	{
		std::cout << "n should be twice as m for now." << std::endl;
		return 1;
	}
	//malloc input data
	std::complex<float> *CPU_A = (std::complex<float>*)malloc(m*n*batchSize*sizeof(std::complex<float>));
	//temperay buffer to hold the intermediate result after the first kernel
	std::complex<float> *CPU_A_TEMP = (std::complex<float>*)malloc(m*n*batchSize*sizeof(std::complex<float>));

	std::complex<float> *CPU_A_OUT = (std::complex<float>*)malloc(m*n*batchSize*sizeof(std::complex<float>));
	int miniBatchSize = n / m;//which is 2 for now
	for (int k = 0; k < batchSize; k++)
	{
		for (int q = 0; q < miniBatchSize; q++)
		{
			for (int i = 0; i < m; i++)
			{
				for (int j = 0; j < n/2; j++)
				{
					CPU_A[k*m*n + q*n/2 + i*n + j] = { (float)(i*n + j), (float)(i*n + j) };
				}
			}
		}
	}
	//std::cout << CPU_A[m*n + 1 * n] << std::endl;
	//init OpenCL 
	cl_int err;
	cl_platform_id platform;
	cl_device_id device;
	cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
	cl_context context;
	cl_command_queue queue;
	cl_kernel kernel1, kernel2;
	cl_event event1, event2;
	char *source1, *source2;

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
	source1 = loadFile(KERNEL_SOURCE1);
	assert(source1 != NULL);
	kernel1 = createKernel(source1, context, BUILD_OPTIONS, &err);
	assert(kernel1 != NULL);

	source2 = loadFile(KERNEL_SOURCE2);
	assert(source2 != NULL);
	kernel2 = createKernel(source2, context, BUILD_OPTIONS, &err);
	assert(kernel2 != NULL);


	//launch kernel
	size_t localWorkSize1[1] = { 256 };
	//calculate number of work groups
	//each work group works on a 32 by 32 block
	//the whole matrix has m/32 * n /32 = 32 x 32 blocks
	//the upper triangle of which (including the diagional) is
	//32*(32+1)/2 = 528
	//so the formula is (m/32) * (m/32 + 1) / 2
	int num_wg = (m / 32) * (m / 32 + 1) / 2;
	size_t globalWorkSize1[1] = { batchSize * num_wg * miniBatchSize * 256 };//528 * 256

	err = clSetKernelArg(kernel1, 0, sizeof(cl_mem), &bufA);
	assert(err == CL_SUCCESS);
	/*
	err = clSetKernelArg(kernel, 1, sizeof(cl_uint), &m);
	assert(err == CL_SUCCESS);
	err = clSetKernelArg(kernel, 2, sizeof(cl_uint), &num_wg);
	assert(err == CL_SUCCESS);
	*/

	//second pass kernel sizes
	size_t localWorkSize2[1] = { 256 };
	size_t globalWorkSize2[1] = { batchSize*((n-2)/11)*256 }; // 2046 / 11 = 186; each WG takes care of 11 rows of 1024 long memory; each WI takes care of 44 element
	err = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &bufA);
	assert(err == CL_SUCCESS);

	if (performance_level == 0)
	{
		//check result
		err = clEnqueueNDRangeKernel(queue, kernel1, 1, NULL,
			globalWorkSize1, localWorkSize1, 0, NULL, &event1);
		assert(err == CL_SUCCESS);
		err = clFinish(queue);
		assert(err == CL_SUCCESS);
		err = clEnqueueReadBuffer(queue, bufA, CL_TRUE, 0,
			(batchSize*m*n) * sizeof(*CPU_A_TEMP), CPU_A_TEMP,
			0, NULL, NULL);
		assert(err == CL_SUCCESS);
		err = clFinish(queue);
		assert(err == CL_SUCCESS);
		//second pass
		err = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL,
			globalWorkSize2, localWorkSize2, 0, NULL, &event2);
		assert(err == CL_SUCCESS);
		err = clFinish(queue);

	}
	else if (performance_level == 1)
	{
		//check kernel performance
		err = clEnqueueNDRangeKernel(queue, kernel1, 1, NULL,
			globalWorkSize1, localWorkSize1, 0, NULL, &event1);
		assert(err == CL_SUCCESS);
		clWaitForEvents(1, &event1);
		assert(err == CL_SUCCESS);
		//second pass
		err = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL,
			globalWorkSize2, localWorkSize2, 0, NULL, &event2);
		assert(err == CL_SUCCESS);
		clWaitForEvents(1, &event2);
		assert(err == CL_SUCCESS);

		cl_ulong start1, end1, start2, end2;
		cl_ulong KernelTime1 = 0;
		cl_ulong KernelTime2 = 0;
		int iteration = 20;
		for (int i = 0; i < iteration; i++)
		{
			event1 = NULL;
			event2 = NULL;
			err = clEnqueueNDRangeKernel(queue, kernel1, 1, NULL,
				globalWorkSize1, localWorkSize1, 0, NULL, &event1);

			assert(err == CL_SUCCESS);
			clWaitForEvents(1, &event1);
			assert(err == CL_SUCCESS);

			err = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL,
				globalWorkSize2, localWorkSize2, 0, NULL, &event2);

			assert(err == CL_SUCCESS);
			clWaitForEvents(1, &event2);
			assert(err == CL_SUCCESS);

			err = clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_START,
				sizeof(start1), &start1, NULL);
			err = clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_END,
				sizeof(end1), &end1, NULL);

			err = clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_START,
				sizeof(start2), &start2, NULL);
			err = clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_END,
				sizeof(end2), &end2, NULL);

			KernelTime1 += (end1 - start1);
			KernelTime2 += (end2 - start2);

		}

		//KernelTime is in ns
		size_t peakGBs = 512;
		std::cout << "the first kernel takes " << KernelTime1/iteration << " ns in average." << std::endl;
		std::cout << "the second kernel takes " << KernelTime2/iteration << " ns in average." << std::endl;

		size_t KernelGBs = 2 * sizeof(std::complex<float>) * m * n * batchSize / ((KernelTime1 + KernelTime2) / iteration);
		std::cout << " GBs: " << KernelGBs << " GBs" << std::endl;
		float efficiency = ((float)KernelGBs) / (float)peakGBs;
		std::cout << " efficiency: " << efficiency * 100 << "%" << std::endl;
	}

	//move memory from device to host
	err = clEnqueueReadBuffer(queue, bufA, CL_TRUE, 0,
		(batchSize*m*n) * sizeof(*CPU_A_OUT), CPU_A_OUT,
		0, NULL, NULL);
	assert(err == CL_SUCCESS);

	if (performance_level == 0)
	{
		//check result
		int error = 0;

		std::cout << CPU_A[m*n + 1] << std::endl;
		std::cout << CPU_A_OUT[m*n + m] << std::endl;
		for (int k = 0; k < batchSize; k++)
		{
			for (int q = 0; q < miniBatchSize; q++)
			{
				for (int i = 0; i < m; i++)
				{
					for (int j = 0; j < n / 2; j++)
					{
						std::complex<float> temp = CPU_A_TEMP[k*m*n + q*n/2 + i*n + j];
						if (CPU_A_TEMP[k*m*n + q*n / 2 + i*n + j].real() != (j*n + i) || CPU_A_TEMP[k*m*n + q*n / 2 + i*n + j].imag() != (j*n + i))
						{
							std::cout << "batch idex = " << k << " ";
							std::cout << "CPU_A[" << i << "*n + " << j << "] = " << CPU_A_TEMP[k*m*n + q*n / 2 + i*n + j] << std::endl;
							error = 1;
							break;
						}
					}
				}
			}
		}

		if (error == 0)
		{
			std::cout << "first kernel correstness passed." << std::endl;
		}
		else
		{
			std::cout << "first kernel correctness failed." << std::endl;
		}

		for (int k = 0; k < batchSize; k++)
		{
			for (int i = 0; i < m; i++)
			{
				for (int j = 0; j < n; j++)
				{
					std::complex<float> temp = CPU_A_OUT[k*m*n + j*m + i];
					if (CPU_A_OUT[k*m*n + j*m + i].real() != CPU_A[k*m*n + i*n + j].real() || CPU_A_OUT[k*m*n + j*m + i].imag() != CPU_A[k*m*n + i*n + j].imag())
					{
						std::cout << "batch idex = " << k << " ";
						std::cout << "CPU_A_OUT[" << j << "*m + " << i << "] = " << CPU_A_OUT[k*m*n + j*m + i] << std::endl;
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
	err = clReleaseEvent(event1);
	err = clReleaseEvent(event2);
	err = clReleaseKernel(kernel1);
	err = clReleaseKernel(kernel2);
	err = clReleaseCommandQueue(queue);
	err = clReleaseContext(context);
	free(CPU_A_TEMP);
	free(CPU_A);
	free(CPU_A_OUT);
}
