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

__constant int swap_table[313][1] = {
{1},
{2},
{4},
{5},
{7},
{8},
{10},
{11},
{13},
{14},
{16},
{17},
{19},
{20},
{22},
{23},
{25},
{26},
{28},
{29},
{31},
{32},
{34},
{35},
{37},
{38},
{40},
{41},
{43},
{44},
{46},
{47},
{49},
{50},
{52},
{53},
{55},
{56},
{58},
{59},
{61},
{62},
{64},
{65},
{67},
{68},
{70},
{71},
{73},
{74},
{76},
{77},
{79},
{80},
{85},
{86},
{88},
{89},
{91},
{92},
{94},
{95},
{97},
{98},
{100},
{101},
{103},
{104},
{106},
{107},
{110},
{112},
{113},
{115},
{116},
{118},
{119},
{121},
{122},
{124},
{125},
{127},
{128},
{130},
{131},
{133},
{134},
{137},
{139},
{140},
{142},
{143},
{145},
{146},
{148},
{149},
{151},
{152},
{154},
{155},
{157},
{158},
{160},
{161},
{169},
{170},
{172},
{173},
{175},
{176},
{178},
{179},
{181},
{182},
{184},
{185},
{187},
{188},
{193},
{194},
{196},
{197},
{199},
{200},
{202},
{203},
{205},
{206},
{208},
{209},
{211},
{212},
{214},
{215},
{220},
{221},
{223},
{224},
{226},
{227},
{229},
{230},
{232},
{233},
{235},
{236},
{238},
{239},
{241},
{242},
{274},
{275},
{277},
{278},
{281},
{283},
{284},
{286},
{287},
{290},
{292},
{293},
{295},
{296},
{301},
{302},
{304},
{305},
{308},
{310},
{311},
{313},
{314},
{317},
{319},
{320},
{322},
{323},
{337},
{338},
{340},
{341},
{344},
{346},
{347},
{349},
{350},
{356},
{358},
{359},
{362},
{364},
{365},
{367},
{368},
{371},
{373},
{374},
{376},
{377},
{383},
{385},
{386},
{389},
{391},
{392},
{394},
{395},
{398},
{400},
{401},
{403},
{404},
{421},
{422},
{425},
{427},
{428},
{430},
{431},
{439},
{440},
{443},
{445},
{446},
{448},
{449},
{452},
{454},
{455},
{457},
{458},
{466},
{467},
{470},
{472},
{473},
{475},
{476},
{479},
{481},
{482},
{484},
{485},
{547},
{548},
{553},
{554},
{556},
{557},
{562},
{563},
{565},
{566},
{589},
{590},
{592},
{593},
{602},
{607},
{608},
{610},
{611},
{616},
{617},
{619},
{620},
{629},
{634},
{635},
{637},
{638},
{643},
{644},
{646},
{647},
{673},
{674},
{688},
{689},
{691},
{692},
{697},
{698},
{700},
{701},
{715},
{716},
{718},
{719},
{724},
{725},
{727},
{728},
{1093},
{1094},
{1097},
{1103},
{1106},
{1121},
{1124},
{1130},
{1133},
{1178},
{1184},
{1187},
{1205},
{1211},
{1214},
{1367},
{1376},
{1430},
{1457}
};

__attribute__(( reqd_work_group_size( 256, 1, 1 ) ))
kernel void
swap_nonsquare( global float2* restrict inputA )
{
    //729 x (3 x 729) x batchSize
	//each wg handles one row of 729 in memory
	const int num_wg_per_batch = 313;
	int group_id = get_group_id(0);
	int idx = get_local_id(0);
	
	int batch_offset = group_id / num_wg_per_batch;
	inputA += batch_offset*1594323;
	group_id -= batch_offset*313;
	
	int prev = swap_table[group_id][0];
	int next = 0;
	
    __local float2 prevValue[729];
	__local float2 nextValue[729];
	
	int group_offset = (prev/3)*729*3 + (prev%3)*729;
	
	prevValue[idx] = inputA[group_offset+idx];
	prevValue[idx+256] = inputA[group_offset + idx + 256 ];
	if(idx + 256*2 < 729)//512-728
	{
	    prevValue[idx+256*2] = inputA[group_offset + idx + 256*2 ];
	}
	barrier(CLK_LOCAL_MEM_FENCE);	
	
	do
	{
		next = (prev*729)%2186;
		group_offset = (next/3)*729*3 + (next%3)*729;
		
	    nextValue[idx] = inputA[group_offset+idx];
	    nextValue[idx+256] = inputA[group_offset + idx + 256 ];
		if(idx + 256*2 < 729)//0-728
		{
			nextValue[idx+256*2] = inputA[group_offset + idx + 256*2 ];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		
	    inputA[group_offset+idx] = prevValue[idx];
	    inputA[group_offset + idx + 256 ] = prevValue[idx+256];

		if(idx + 256*2 < 729)//0-728
		{
			inputA[group_offset + idx + 256*2 ] = prevValue[idx+256*2];
		}
	    barrier(CLK_LOCAL_MEM_FENCE);
		
		prevValue[idx] = nextValue[idx];
		prevValue[idx+256] = nextValue[idx+256];
		if(idx + 256*2 < 729)//0-728
		{
			prevValue[idx+256*2] = nextValue[idx + 256*2 ];
		}
	    barrier(CLK_LOCAL_MEM_FENCE);		
		
		prev = next;
	}while(next!=swap_table[group_id][0]);
}