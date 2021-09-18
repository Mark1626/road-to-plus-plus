#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

__kernel void cl_stabilize(
	__global const unsigned char* input,
	__global unsigned char* output,
	__global int* spills,
	const int height,
	const int width) {
	int lid = get_local_id(0);
	int lsize = get_local_size(0);
	local int localSpills[1];
	if(lid == 0){
		localSpills[0]=0;
	}
	int i = get_global_id(0);

	bool outOfBounds = (i < width);
	outOfBounds |= (i > (width * (height - 1)));
	outOfBounds |= (i % width == 0);
	outOfBounds |= (i % width == width-1);

	if (outOfBounds) { output[i] = 0; return; }

	int pixelup = i - width;
	int pixeldown = i + width;
	int pixelleft = i - 1;
	int pixelright = i + 1;

	int currSand = input[i];
	int spill = currSand >= 4;
	int newSand = currSand >= 4 ? currSand - 4 : currSand;

	if (spill) {
		barrier(CLK_LOCAL_MEM_FENCE);
			atomic_add(&localSpills[0], 1);
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (lid == lsize - 1) {
		// int total_sum = 0;
		// for (int idx = 0; i < 4; i++) {
		// 	total_sum += localSpills[idx];
		// }
		// atomic_add(spills, total_sum);
		atomic_add(spills, localSpills[0]);
	}

	newSand = newSand + (input[pixelup] >= 4);
	newSand = newSand + (input[pixeldown] >= 4);
	newSand = newSand + (input[pixelleft] >= 4);
	newSand = newSand + (input[pixelright] >= 4);

	output[i] = newSand;
}