__constant int swap_table[6][1] = {
{8}, 
{3}, 
{5},
{25},
{26},
{27}
};
__attribute__(( reqd_work_group_size( 256, 1, 1 ) ))
kernel void
swap_nonsquare( global float* restrict inputA )
{
    int idx = get_global_id(0);
	int prev = swap_table[idx][0];
	
	int next = 0;
	float prevValue = inputA[prev];
	float nextValue;
	do
	{
	    next = (prev * 4) % 31;
	    nextValue = inputA[next];
		inputA[next] = prevValue;
		prevValue = nextValue;
		prev = next;
	}while(next!=swap_table[idx][0]);
}