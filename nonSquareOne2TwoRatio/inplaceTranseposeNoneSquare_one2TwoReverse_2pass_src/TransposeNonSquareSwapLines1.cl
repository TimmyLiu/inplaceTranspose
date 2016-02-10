/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/








__constant int swap_table[186][1] = {
{1},
{3},
{5},
{7},
{9},
{11},
{13},
{15},
{17},
{19},
{21},
{23},
{25},
{27},
{29},
{31},
{33},
{35},
{37},
{39},
{41},
{43},
{45},
{47},
{49},
{51},
{53},
{55},
{57},
{59},
{61},
{63},
{67},
{69},
{71},
{73},
{75},
{77},
{79},
{81},
{83},
{85},
{87},
{89},
{91},
{93},
{95},
{99},
{101},
{103},
{105},
{107},
{109},
{111},
{113},
{115},
{117},
{119},
{121},
{123},
{125},
{127},
{137},
{139},
{141},
{143},
{147},
{149},
{151},
{153},
{155},
{157},
{159},
{163},
{165},
{167},
{169},
{171},
{173},
{175},
{179},
{181},
{183},
{185},
{187},
{189},
{191},
{199},
{201},
{203},
{205},
{207},
{211},
{213},
{215},
{217},
{219},
{221},
{223},
{229},
{231},
{233},
{235},
{237},
{239},
{243},
{245},
{247},
{249},
{251},
{253},
{255},
{293},
{295},
{299},
{301},
{303},
{307},
{309},
{311},
{315},
{317},
{319},
{331},
{333},
{335},
{339},
{341},
{343},
{347},
{349},
{351},
{359},
{363},
{365},
{367},
{371},
{373},
{375},
{379},
{381},
{383},
{411},
{413},
{415},
{423},
{427},
{429},
{431},
{437},
{439},
{443},
{445},
{447},
{463},
{469},
{471},
{475},
{477},
{479},
{491},
{493},
{495},
{501},
{503},
{507},
{509},
{511},
{683},
{687},
{695},
{699},
{703},
{727},
{731},
{735},
{751},
{759},
{763},
{767},
{879},
{887},
{895},
{959},
{991},
{1023}
};

__attribute__(( reqd_work_group_size( 256, 1, 1 ) ))
kernel void
swap_nonsquare( global float2* restrict inputA )
{
	const int num_wg_per_batch = 186;
	int group_id = get_group_id(0);
	int idx = get_local_id(0);
	
	int batch_offset = group_id / num_wg_per_batch;
	inputA += batch_offset*2097152;
	group_id -= batch_offset*186;
	
	int prev = swap_table[group_id][0];
	int next = 0;
	
    __local float2 prevValue[1024];
	__local float2 nextValue[1024];
	
	int group_offset = (prev/2)*2048 + (prev%2)*1024;
	
	prevValue[idx] = inputA[group_offset+idx];
	prevValue[idx+256] = inputA[group_offset + idx + 256 ];
	prevValue[idx+256*2] = inputA[group_offset + idx + 256*2 ];
	prevValue[idx+256*3] = inputA[group_offset + idx + 256*3 ];
	barrier(CLK_LOCAL_MEM_FENCE);	
	do
	{
		next = (prev*1024)%2047;
		group_offset = (next/2)*2048 + (next%2)*1024;
		
	    nextValue[idx] = inputA[group_offset+idx];
	    nextValue[idx+256] = inputA[group_offset + idx + 256 ];
	    nextValue[idx+256*2] = inputA[group_offset + idx + 256*2 ];
	    nextValue[idx+256*3] = inputA[group_offset + idx + 256*3 ];
		barrier(CLK_LOCAL_MEM_FENCE);
		
	    inputA[group_offset+idx] = prevValue[idx];
	    inputA[group_offset + idx + 256 ] = prevValue[idx+256];
	    inputA[group_offset + idx + 256*2 ] = prevValue[idx+256*2];
	    inputA[group_offset + idx + 256*3 ] = prevValue[idx+256*3];
	    barrier(CLK_LOCAL_MEM_FENCE);
		
		prevValue[idx] = nextValue[idx];
		prevValue[idx+256] = nextValue[idx+256];
		prevValue[idx+256*2] = nextValue[idx+256*2];
		prevValue[idx+256*3] = nextValue[idx+256*3];
	    barrier(CLK_LOCAL_MEM_FENCE);		
		
		prev = next;
	}while(next!=swap_table[group_id][0]);
}