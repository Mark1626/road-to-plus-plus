__kernel void cl_gol(
	__global const bool* input,
	__global bool* output,
	const int height,
	const int width
) {
	int i = get_global_id(0);
	int rowup = i - width;
	int rowdown = i + width;

	bool outOfBounds = (i < width);
	outOfBounds |= (i > (width * (height - 1)));
	outOfBounds |= (i % width == 0);
	outOfBounds |= (i % width == width-1);

	if (outOfBounds) { output[i] = false; return; }

	int neighbours = input[rowup-1] + input[rowup] + input[rowup+1];
	neighbours += input[i-1] + input[i+1];
	neighbours += input[rowdown-1] + input[rowdown] + input[rowdown+1];

	if (neighbours == 3 || (input[i] && neighbours == 2)) {
		output[i] = true;
	} else {
		output[i] = false;
	}
}
